/*
 * SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
 * SPDX-License-Identifier: Apache-2.0
 *
 * Minimal Wayland/EGL display library for rendering DMA-BUF RGBA textures.
 * Designed to be called from Python via ctypes.
 *
 * API:
 *   void *display_init(int width, int height, const char *title);
 *   int   display_render_dmabuf(void *d, int dmabuf_fd, int w, int h);
 *   int   display_render_pixels(void *d, const void *rgba, int w, int h);
 *   int   display_is_open(void *d);
 *   void  display_destroy(void *d);
 *
 * Build (shared library):
 *   gcc -shared -fPIC -o libwlegl_display.so wlegl_display.c \
 *       -lwayland-client -lwayland-egl -lEGL -lGLESv2 -DLINUX -DWL_EGL_PLATFORM
 *
 * Build (standalone test):
 *   gcc -DSTANDALONE -o wlegl_test wlegl_display.c \
 *       -lwayland-client -lwayland-egl -lEGL -lGLESv2 -DLINUX -DWL_EGL_PLATFORM
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/dma-buf.h>

#include <wayland-client.h>
#include <wayland-egl.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

/* ── xdg-shell protocol inline definitions ─────────────────────────────
 *
 * Generated from xdg-shell.xml v6. We inline the wl_interface structs
 * so the library compiles without wayland-scanner on the target.
 * Only the interfaces we actually use are defined.
 */

static const struct wl_interface *xdg_types[4] = { NULL, NULL, NULL, NULL };

/* xdg_wm_base: requests[0..3] = destroy, create_positioner,
 *              get_xdg_surface, pong; events[0] = ping */
static const struct wl_message xdg_wm_base_requests[] = {
    { "destroy",           "",   xdg_types + 0 },
    { "create_positioner", "n",  xdg_types + 0 },
    { "get_xdg_surface",   "no", xdg_types + 0 },
    { "pong",              "u",  xdg_types + 0 },
};
static const struct wl_message xdg_wm_base_events[] = {
    { "ping", "u", xdg_types + 0 },
};
static const struct wl_interface xdg_wm_base_interface = {
    "xdg_wm_base", 6,
    4, xdg_wm_base_requests,
    1, xdg_wm_base_events,
};

/* xdg_surface: requests[0..4], events[0] = configure */
static const struct wl_message xdg_surface_requests[] = {
    { "destroy",             "",     xdg_types + 0 },
    { "get_toplevel",        "n",    xdg_types + 0 },
    { "get_popup",           "n?oo", xdg_types + 0 },
    { "set_window_geometry", "iiii", xdg_types + 0 },
    { "ack_configure",       "u",    xdg_types + 0 },
};
static const struct wl_message xdg_surface_events[] = {
    { "configure", "u", xdg_types + 0 },
};
static const struct wl_interface xdg_surface_interface = {
    "xdg_surface", 6,
    5, xdg_surface_requests,
    1, xdg_surface_events,
};

/* xdg_toplevel: requests[0..13], events[0..3] */
static const struct wl_message xdg_toplevel_requests[] = {
    { "destroy",          "",    xdg_types + 0 },
    { "set_parent",       "?o",  xdg_types + 0 },
    { "set_title",        "s",   xdg_types + 0 },
    { "set_app_id",       "s",   xdg_types + 0 },
    { "show_window_menu", "ouii",xdg_types + 0 },
    { "move",             "ou",  xdg_types + 0 },
    { "resize",           "ouu", xdg_types + 0 },
    { "set_max_size",     "ii",  xdg_types + 0 },
    { "set_min_size",     "ii",  xdg_types + 0 },
    { "set_maximized",    "",    xdg_types + 0 },
    { "unset_maximized",  "",    xdg_types + 0 },
    { "set_fullscreen",   "?o",  xdg_types + 0 },
    { "unset_fullscreen", "",    xdg_types + 0 },
    { "set_minimized",    "",    xdg_types + 0 },
};
static const struct wl_message xdg_toplevel_events[] = {
    { "configure",        "iia", xdg_types + 0 },
    { "close",            "",    xdg_types + 0 },
    { "configure_bounds", "ii",  xdg_types + 0 },
    { "wm_capabilities",  "a",   xdg_types + 0 },
};
static const struct wl_interface xdg_toplevel_interface = {
    "xdg_toplevel", 6,
    14, xdg_toplevel_requests,
    4,  xdg_toplevel_events,
};

/* ── EGL extension function pointers ───────────────────────────────────── */

static PFNEGLGETPLATFORMDISPLAYEXTPROC  fn_eglGetPlatformDisplayEXT;
static PFNEGLCREATEIMAGEKHRPROC         fn_eglCreateImageKHR;
static PFNEGLDESTROYIMAGEKHRPROC        fn_eglDestroyImageKHR;
static PFNGLEGLIMAGETARGETTEXTURE2DOESPROC fn_glEGLImageTargetTexture2DOES;

/* ── Shaders ───────────────────────────────────────────────────────────── */

static const char *vert_src =
    "attribute vec2 pos;\n"
    "attribute vec2 uv;\n"
    "varying vec2 v_uv;\n"
    "void main() {\n"
    "  gl_Position = vec4(pos, 0.0, 1.0);\n"
    "  v_uv = uv;\n"
    "}\n";

static const char *frag_src =
    "precision mediump float;\n"
    "varying vec2 v_uv;\n"
    "uniform sampler2D tex;\n"
    "void main() {\n"
    "  gl_FragColor = texture2D(tex, v_uv);\n"
    "}\n";

/* Fullscreen quad: pos(x,y) + uv(s,t). UVs flipped vertically. */
static const float quad_verts[] = {
    -1, -1,  0, 1,
     1, -1,  1, 1,
    -1,  1,  0, 0,
     1,  1,  1, 0,
};

/* ── Display state ─────────────────────────────────────────────────────── */

#define MAX_DMABUF_SLOTS 4

typedef struct {
    int         fd;
    EGLImageKHR image;
} dmabuf_slot_t;

typedef struct {
    struct wl_display    *wl_display;
    struct wl_registry   *wl_registry;
    struct wl_compositor *wl_compositor;
    struct wl_surface    *wl_surface;
    struct wl_proxy      *xdg_wm_base;
    struct wl_proxy      *xdg_surface;
    struct wl_proxy      *xdg_toplevel;
    struct wl_egl_window *egl_window;
    int configured, closed;

    EGLDisplay  egl_display;
    EGLContext  egl_context;
    EGLSurface  egl_surface;

    GLuint program, texture;
    GLint  pos_loc, uv_loc, tex_loc;

    dmabuf_slot_t slots[MAX_DMABUF_SLOTS];
    int           n_slots;
    int width, height;
} display_t;

/* ── Wayland event handlers ────────────────────────────────────────────── */

static void handle_wm_base_ping(void *data, struct wl_proxy *p, uint32_t serial) {
    (void)data;
    wl_proxy_marshal(p, 3, serial); /* pong opcode=3 */
}

static void handle_surface_configure(void *data, struct wl_proxy *p, uint32_t serial) {
    display_t *d = data;
    wl_proxy_marshal(p, 4, serial); /* ack_configure opcode=4 */
    d->configured = 1;
}

static void handle_toplevel_configure(void *data, struct wl_proxy *p,
                                       int32_t w, int32_t h, struct wl_array *s) {
    (void)data; (void)p; (void)w; (void)h; (void)s;
}
static void handle_toplevel_close(void *data, struct wl_proxy *p) {
    ((display_t *)data)->closed = 1; (void)p;
}
static void handle_toplevel_bounds(void *data, struct wl_proxy *p, int32_t w, int32_t h) {
    (void)data; (void)p; (void)w; (void)h;
}
static void handle_toplevel_caps(void *data, struct wl_proxy *p, struct wl_array *a) {
    (void)data; (void)p; (void)a;
}

static const void *wm_base_listener[]  = { (void *)handle_wm_base_ping };
static const void *surface_listener[]  = { (void *)handle_surface_configure };
static const void *toplevel_listener[] = {
    (void *)handle_toplevel_configure,
    (void *)handle_toplevel_close,
    (void *)handle_toplevel_bounds,
    (void *)handle_toplevel_caps,
};

/* ── Registry ──────────────────────────────────────────────────────────── */

static void reg_global(void *data, struct wl_registry *r,
                       uint32_t name, const char *iface, uint32_t ver) {
    display_t *d = data;
    if (strcmp(iface, "wl_compositor") == 0)
        d->wl_compositor = wl_registry_bind(r, name, &wl_compositor_interface,
                                             ver < 4 ? ver : 4);
    else if (strcmp(iface, "xdg_wm_base") == 0)
        d->xdg_wm_base = (struct wl_proxy *)wl_registry_bind(
            r, name, &xdg_wm_base_interface, 1);
}
static void reg_global_remove(void *data, struct wl_registry *r, uint32_t n) {
    (void)data; (void)r; (void)n;
}
static const struct wl_registry_listener reg_listener = { reg_global, reg_global_remove };

/* ── GL helpers ────────────────────────────────────────────────────────── */

static GLuint compile_shader(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(s, sizeof(log), NULL, log);
        fprintf(stderr, "display: shader: %s\n", log);
    }
    return s;
}

static GLuint build_program(void) {
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vert_src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, frag_src);
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(p, sizeof(log), NULL, log);
        fprintf(stderr, "display: link: %s\n", log);
    }
    return p;
}

static void draw_quad(display_t *d) {
    glViewport(0, 0, d->width, d->height);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(d->program);

    glVertexAttribPointer(d->pos_loc, 2, GL_FLOAT, GL_FALSE, 16, quad_verts);
    glEnableVertexAttribArray(d->pos_loc);
    glVertexAttribPointer(d->uv_loc, 2, GL_FLOAT, GL_FALSE, 16, quad_verts + 2);
    glEnableVertexAttribArray(d->uv_loc);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, d->texture);
    glUniform1i(d->tex_loc, 0);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glFlush();
    eglSwapBuffers(d->egl_display, d->egl_surface);
}

/* ── DMA-BUF cache coherency ─────────────────────────────────────────── */

static void dmabuf_sync(int fd, int start, int flags) {
    struct dma_buf_sync sync = {
        .flags = (uint64_t)((start ? DMA_BUF_SYNC_START : DMA_BUF_SYNC_END) | flags)
    };
    if (ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync) < 0)
        perror("display: DMA_BUF_IOCTL_SYNC");
}

/* ── Public API ────────────────────────────────────────────────────────── */

void *display_init(int width, int height, const char *title) {
    display_t *d = calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->width = width;
    d->height = height;
    d->n_slots = 0;
    for (int i = 0; i < MAX_DMABUF_SLOTS; i++) {
        d->slots[i].fd = -1;
        d->slots[i].image = EGL_NO_IMAGE_KHR;
    }

    /* EGL extensions */
    fn_eglGetPlatformDisplayEXT =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    fn_eglCreateImageKHR =
        (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
    fn_eglDestroyImageKHR =
        (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImageKHR");
    fn_glEGLImageTargetTexture2DOES =
        (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress("glEGLImageTargetTexture2DOES");

    /* 1. Connect to Wayland */
    d->wl_display = wl_display_connect(NULL);
    if (!d->wl_display) {
        fprintf(stderr, "display: no Wayland compositor\n");
        free(d);
        return NULL;
    }

    d->wl_registry = wl_display_get_registry(d->wl_display);
    wl_registry_add_listener(d->wl_registry, &reg_listener, d);
    wl_display_roundtrip(d->wl_display);

    if (!d->wl_compositor || !d->xdg_wm_base) {
        fprintf(stderr, "display: missing wl_compositor or xdg_wm_base\n");
        goto fail;
    }

    /* 2. Setup xdg shell objects */
    wl_proxy_add_listener(d->xdg_wm_base, (void (**)(void))wm_base_listener, d);

    d->wl_surface = wl_compositor_create_surface(d->wl_compositor);

    d->xdg_surface = wl_proxy_marshal_constructor(
        d->xdg_wm_base, 2, &xdg_surface_interface, NULL, d->wl_surface);
    wl_proxy_add_listener(d->xdg_surface, (void (**)(void))surface_listener, d);

    d->xdg_toplevel = wl_proxy_marshal_constructor(
        d->xdg_surface, 1, &xdg_toplevel_interface, NULL);
    wl_proxy_add_listener(d->xdg_toplevel, (void (**)(void))toplevel_listener, d);

    if (title)
        wl_proxy_marshal(d->xdg_toplevel, 2, title);
    wl_proxy_marshal(d->xdg_toplevel, 3, "ara2-demo");

    wl_surface_commit(d->wl_surface);

    while (!d->configured)
        wl_display_dispatch(d->wl_display);

    /* 3. EGL */
    d->egl_window = wl_egl_window_create(d->wl_surface, width, height);

    if (fn_eglGetPlatformDisplayEXT)
        d->egl_display = fn_eglGetPlatformDisplayEXT(
            EGL_PLATFORM_WAYLAND_KHR, d->wl_display, NULL);
    else
        d->egl_display = eglGetDisplay((EGLNativeDisplayType)d->wl_display);

    EGLint major, minor;
    if (!eglInitialize(d->egl_display, &major, &minor)) {
        fprintf(stderr, "display: eglInitialize failed\n");
        goto fail;
    }
    eglBindAPI(EGL_OPENGL_ES_API);

    EGLint cfg_attr[] = {
        EGL_SURFACE_TYPE,    EGL_WINDOW_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_NONE
    };
    EGLConfig config;
    EGLint n_cfg;
    eglChooseConfig(d->egl_display, cfg_attr, &config, 1, &n_cfg);

    EGLint ctx_attr[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
    d->egl_context = eglCreateContext(d->egl_display, config,
                                       EGL_NO_CONTEXT, ctx_attr);
    d->egl_surface = eglCreateWindowSurface(d->egl_display, config,
                                             (EGLNativeWindowType)d->egl_window, NULL);
    eglMakeCurrent(d->egl_display, d->egl_surface, d->egl_surface, d->egl_context);

    /* Disable vsync throttling — render as fast as the pipeline allows.
       Without this, eglSwapBuffers blocks on compositor frame callbacks,
       adding up to 16.7ms per frame of display latency. */
    eglSwapInterval(d->egl_display, 0);

    /* 4. GL setup */
    d->program = build_program();
    d->pos_loc = glGetAttribLocation(d->program, "pos");
    d->uv_loc  = glGetAttribLocation(d->program, "uv");
    d->tex_loc = glGetUniformLocation(d->program, "tex");

    glGenTextures(1, &d->texture);
    glBindTexture(GL_TEXTURE_2D, d->texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    fprintf(stderr, "display: %dx%d EGL %d.%d (%s)\n",
            width, height, major, minor, glGetString(GL_RENDERER));
    return d;

fail:
    if (d->wl_display) wl_display_disconnect(d->wl_display);
    free(d);
    return NULL;
}

int display_render_dmabuf(void *ptr, int dmabuf_fd, int width, int height) {
    display_t *d = ptr;
    if (!d || !fn_eglCreateImageKHR) return -1;

    wl_display_dispatch_pending(d->wl_display);
    wl_display_flush(d->wl_display);
    if (d->closed) return -2;

    /* Look up cached EGLImage for this fd, or create one.
       Slots are populated once at startup and reused every frame. */
    EGLImageKHR image = EGL_NO_IMAGE_KHR;
    for (int i = 0; i < d->n_slots; i++) {
        if (d->slots[i].fd == dmabuf_fd) {
            image = d->slots[i].image;
            break;
        }
    }

    if (image == EGL_NO_IMAGE_KHR) {
        if (d->n_slots >= MAX_DMABUF_SLOTS) {
            fprintf(stderr, "display: too many DMA-BUF slots\n");
            return -1;
        }

        EGLint attrs[] = {
            EGL_WIDTH,                      width,
            EGL_HEIGHT,                     height,
            EGL_LINUX_DRM_FOURCC_EXT,       0x34324241, /* DRM_FORMAT_ABGR8888 */
            EGL_DMA_BUF_PLANE0_FD_EXT,     dmabuf_fd,
            EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0,
            EGL_DMA_BUF_PLANE0_PITCH_EXT,  width * 4,
            EGL_NONE
        };

        image = fn_eglCreateImageKHR(
            d->egl_display, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, attrs);
        if (image == EGL_NO_IMAGE_KHR) {
            fprintf(stderr, "display: eglCreateImageKHR failed 0x%x\n", eglGetError());
            return -1;
        }

        d->slots[d->n_slots].fd = dmabuf_fd;
        d->slots[d->n_slots].image = image;
        d->n_slots++;
        fprintf(stderr, "display: cached EGLImage for fd %d (slot %d)\n",
                dmabuf_fd, d->n_slots - 1);
    }

    /* Ensure the DMA-BUF contents written by HAL's headless EGL context
       are visible to this context's GPU texture cache before reading. */
    dmabuf_sync(dmabuf_fd, 1, DMA_BUF_SYNC_READ);

    glBindTexture(GL_TEXTURE_2D, d->texture);
    fn_glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, image);

    draw_quad(d);

    dmabuf_sync(dmabuf_fd, 0, DMA_BUF_SYNC_READ);
    return 0;
}

int display_render_pixels(void *ptr, const void *rgba, int width, int height) {
    display_t *d = ptr;
    if (!d || !rgba) return -1;

    wl_display_dispatch_pending(d->wl_display);
    wl_display_flush(d->wl_display);
    if (d->closed) return -2;

    glBindTexture(GL_TEXTURE_2D, d->texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, rgba);

    draw_quad(d);
    return 0;
}

int display_is_open(void *ptr) {
    display_t *d = ptr;
    if (!d) return 0;
    wl_display_dispatch_pending(d->wl_display);
    wl_display_flush(d->wl_display);
    return !d->closed;
}

void display_destroy(void *ptr) {
    display_t *d = ptr;
    if (!d) return;

    if (fn_eglDestroyImageKHR) {
        for (int i = 0; i < d->n_slots; i++) {
            if (d->slots[i].image != EGL_NO_IMAGE_KHR)
                fn_eglDestroyImageKHR(d->egl_display, d->slots[i].image);
        }
    }
    if (d->texture)  glDeleteTextures(1, &d->texture);
    if (d->program)  glDeleteProgram(d->program);

    eglMakeCurrent(d->egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (d->egl_surface) eglDestroySurface(d->egl_display, d->egl_surface);
    if (d->egl_context) eglDestroyContext(d->egl_display, d->egl_context);
    if (d->egl_display) eglTerminate(d->egl_display);

    if (d->egl_window)    wl_egl_window_destroy(d->egl_window);
    if (d->xdg_toplevel)  wl_proxy_destroy(d->xdg_toplevel);
    if (d->xdg_surface)   wl_proxy_destroy(d->xdg_surface);
    if (d->wl_surface)    wl_surface_destroy(d->wl_surface);
    if (d->xdg_wm_base)   wl_proxy_destroy(d->xdg_wm_base);
    if (d->wl_compositor) wl_compositor_destroy(d->wl_compositor);
    if (d->wl_registry)   wl_registry_destroy(d->wl_registry);
    if (d->wl_display)    wl_display_disconnect(d->wl_display);

    free(d);
}

/* ── Standalone test ───────────────────────────────────────────────────── */

#ifdef STANDALONE
int main(void) {
    void *d = display_init(640, 480, "EGL Test");
    if (!d) { fprintf(stderr, "init failed\n"); return 1; }

    int npx = 640 * 480;
    unsigned char *px = calloc(npx * 4, 1);
    for (int i = 0; i < npx; i++) {
        px[i * 4 + 1] = 255;  /* green */
        px[i * 4 + 3] = 255;  /* opaque */
    }

    printf("Rendering green for 4 seconds...\n");
    for (int f = 0; f < 120 && display_is_open(d); f++) {
        display_render_pixels(d, px, 640, 480);
        usleep(33333);
    }

    free(px);
    display_destroy(d);
    printf("Done.\n");
    return 0;
}
#endif
