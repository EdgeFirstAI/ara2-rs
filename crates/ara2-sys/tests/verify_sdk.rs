//! Checks ara2-sys's DVAPI declarations against a real SDK's headers.
//!
//! For each version whose SDK include directory is named by an environment
//! variable (`ARA2_SDK_V1_1`, `ARA2_SDK_V1_3`, `ARA2_SDK_V3_0`), this
//! builds a C translation unit that includes the SDK's `dvapi.h`, asserts
//! every size, alignment, field offset and enumerator value ara2-sys
//! declares, checks each enum typedef's size and signedness, and then
//! redeclares every function prototype ara2-sys binds, so a mismatched
//! signature is a "conflicting types" compile error. Versions with no
//! variable set are skipped, which is the normal case in CI, unless
//! `ARA2_SDK_REQUIRE` lists them (comma-separated versions, or `all`).

use std::{
    fmt::Write as _,
    path::{Path, PathBuf},
    process::Command,
};

struct Version {
    name: &'static str,
    env: &'static str,
    header: &'static str,
    bindings: &'static str,
}

const VERSIONS: &[Version] = &[
    Version {
        name: "1.1",
        env: "ARA2_SDK_V1_1",
        header: "include/dvapi_v1_1.h",
        bindings: "src/v1_1.rs",
    },
    Version {
        name: "1.3",
        env: "ARA2_SDK_V1_3",
        header: "include/dvapi_v1_3.h",
        bindings: "src/v1_3.rs",
    },
    Version {
        name: "3.0",
        env: "ARA2_SDK_V3_0",
        header: "include/dvapi_v3_0.h",
        bindings: "src/v3_0.rs",
    },
];

const COMMON_HEADER: &str = "include/dvapi_common.h";
const COMMON_BINDINGS: &str = "src/ffi.rs";

fn crate_file(rel: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// `(label, value)` from bindgen's layout assertions, e.g.
/// `("Offset of field: dv_model::num_inputs", 56)`. Panics on anything it
/// cannot parse so a changed bindgen format cannot shrink the checked set.
fn layout_facts(bindings: &str) -> Vec<(String, u64)> {
    let mut facts = Vec::new();
    let mut rest = bindings;
    while let Some(start) = rest.find("[\"") {
        rest = &rest[start + 2..];
        let end = rest.find("\"]").expect("unterminated layout label");
        let label = rest[..end].to_owned();
        rest = &rest[end + 2..];
        let minus = rest
            .find(" - ")
            .unwrap_or_else(|| panic!("no ' - ' after layout label {label:?}"));
        let close = rest
            .find("usize]")
            .unwrap_or_else(|| panic!("no 'usize]' after layout label {label:?}"));
        assert!(
            minus < close,
            "layout label {label:?} is followed by another layout fact before its value"
        );
        let text = rest[minus + 3..close].trim();
        let value = text
            .parse()
            .unwrap_or_else(|_| panic!("layout value {text:?} for {label:?}"));
        facts.push((label, value));
        rest = &rest[close + 6..];
    }
    facts
}

/// The C name of a field bindgen renamed to dodge a Rust keyword.
fn c_field(name: &str) -> &str {
    match name {
        "type_" => "type",
        "ref_" => "ref",
        "async_" => "async",
        other => other,
    }
}

/// A `_Static_assert` for one layout fact, or `None` for types ara2-sys
/// does not declare itself (libc types, bindgen's anonymous unions).
fn layout_assert(label: &str, value: u64) -> Option<String> {
    let (expr, ty) = if let Some(ty) = label.strip_prefix("Size of ") {
        (format!("sizeof({ty}_t)"), ty)
    } else if let Some(ty) = label.strip_prefix("Alignment of ") {
        (format!("_Alignof({ty}_t)"), ty)
    } else if let Some(path) = label.strip_prefix("Offset of field: ") {
        let (ty, field) = path.split_once("::")?;
        (format!("offsetof({ty}_t, {})", c_field(field)), ty)
    } else {
        return None;
    };
    if !ty.starts_with("dv_") || ty.contains("__bindgen") {
        return None;
    }
    Some(format!("_Static_assert({expr} == {value}, \"{label}\");"))
}

fn is_enumerator_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

fn code_part(line: &str) -> &str {
    line.split("/*")
        .next()
        .unwrap_or("")
        .split("//")
        .next()
        .unwrap_or("")
        .trim()
}

/// `(name, value)` for every enumerator in one of our headers. Panics on an
/// `UPPER_NAME = ...` line whose value is not a plain integer.
fn enumerators(header: &str) -> Vec<(String, i64)> {
    header
        .lines()
        .filter_map(|line| {
            let code = code_part(line).trim_end_matches(',');
            let (name, value) = code.split_once('=')?;
            let name = name.trim();
            if !is_enumerator_name(name) {
                return None;
            }
            let parsed = value
                .trim()
                .parse()
                .unwrap_or_else(|_| panic!("enumerator value is not a plain integer: {line:?}"));
            Some((name.to_owned(), parsed))
        })
        .collect()
}

/// Lines holding an `=` inside an `enum { ... }` block, counted
/// independently of `enumerators`.
fn enum_assignment_lines(header: &str) -> usize {
    let mut inside = false;
    let mut count = 0;
    for line in header.lines().map(code_part) {
        if line.contains("enum ") && line.ends_with('{') {
            inside = true;
        } else if inside && line.starts_with('}') {
            inside = false;
        } else if inside && line.contains('=') {
            count += 1;
        }
    }
    count
}

/// `(typedef name, any negative enumerator)` for each `typedef enum`.
fn enum_typedefs(header: &str) -> Vec<(String, bool)> {
    let mut out = Vec::new();
    let mut negative = None;
    for line in header.lines().map(str::trim) {
        if line.starts_with("typedef enum") {
            negative = Some(false);
        } else if let Some(neg) = negative.as_mut() {
            if let Some(name) = line.strip_prefix('}') {
                out.push((name.trim().trim_end_matches(';').trim().to_owned(), *neg));
                negative = None;
            } else if line.contains("= -") {
                *neg = true;
            }
        }
    }
    out
}

/// Whether `line` mentions `dv_<ident>(`.
fn has_dv_call_syntax(line: &str) -> bool {
    line.match_indices("dv_").any(|(i, _)| {
        let tail = &line[i + 3..];
        let ident = tail
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .count();
        ident > 0 && tail[ident..].starts_with('(')
    })
}

/// Every one-line function prototype in one of our headers. Panics on a
/// declaration that spans lines, which the verifier cannot redeclare.
fn prototypes(header: &str) -> Vec<String> {
    header
        .lines()
        .map(str::trim)
        .filter(|l| !l.starts_with('*') && !l.starts_with("/*") && !l.starts_with("//"))
        .map(code_part)
        .filter(|l| has_dv_call_syntax(l))
        .map(|l| {
            assert!(
                l.ends_with(");"),
                "function prototypes must be complete on one line: {l:?}"
            );
            l.to_owned()
        })
        .collect()
}

/// Prototype-looking lines counted without `has_dv_call_syntax`.
fn prototype_lines(header: &str) -> usize {
    header
        .lines()
        .map(str::trim)
        .filter(|l| !l.starts_with('*') && !l.starts_with("/*") && !l.starts_with("//"))
        .filter(|l| l.starts_with("dv_") || l.contains(" dv_") || l.contains("*dv_"))
        .filter(|l| l.contains('(') && l.ends_with(");"))
        .count()
}

fn check_source(version: &Version) -> String {
    let mut src = String::from("#include <stddef.h>\n#include \"dvapi.h\"\n\n");
    for bindings in [crate_file(COMMON_BINDINGS), crate_file(version.bindings)] {
        for (label, value) in layout_facts(&bindings) {
            if let Some(line) = layout_assert(&label, value) {
                writeln!(src, "{line}").unwrap();
            }
        }
    }
    for header in [crate_file(COMMON_HEADER), crate_file(version.header)] {
        for (name, value) in enumerators(&header) {
            writeln!(src, "_Static_assert({name} == {value}, \"{name}\");").unwrap();
        }
        for (name, negative) in enum_typedefs(&header) {
            writeln!(
                src,
                "_Static_assert(sizeof({name}) == 4, \"size of {name}\");"
            )
            .unwrap();
            writeln!(
                src,
                "_Static_assert((({name})-1 < 0) == {}, \"signedness of {name}\");",
                i32::from(negative)
            )
            .unwrap();
        }
        for proto in prototypes(&header) {
            writeln!(src, "{proto}").unwrap();
        }
    }
    src
}

fn verify(version: &Version) {
    let Ok(sdk) = std::env::var(version.env) else {
        let required = std::env::var("ARA2_SDK_REQUIRE").unwrap_or_default();
        assert!(
            !required.split(',').any(|r| r == "all" || r == version.name),
            "{} is not set but ARA2_SDK_REQUIRE names DVAPI {}",
            version.env,
            version.name
        );
        eprintln!(
            "skipping DVAPI {}: {} is not set",
            version.name, version.env
        );
        return;
    };
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let file = dir.join(format!("verify_dvapi_{}.c", version.name.replace('.', "_")));
    std::fs::write(&file, check_source(version)).expect("write check source");
    let output = Command::new("cc")
        .args(["-std=gnu11", "-fsyntax-only", "-I"])
        .arg(&sdk)
        .arg(&file)
        .output()
        .expect("run cc");
    assert!(
        output.status.success(),
        "ara2-sys declarations disagree with the DVAPI {} SDK at {sdk} (source: {}):\n{}",
        version.name,
        file.display(),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn v1_1_matches_sdk() {
    verify(&VERSIONS[0]);
}

#[test]
fn v1_3_matches_sdk() {
    verify(&VERSIONS[1]);
}

#[test]
fn v3_0_matches_sdk() {
    verify(&VERSIONS[2]);
}

#[test]
fn parsers_see_the_declarations() {
    let common = crate_file(COMMON_HEADER);
    assert!(enumerators(&common).contains(&("INVALID_PRODUCT".to_owned(), -1)));
    assert!(enum_typedefs(&common).contains(&("dv_product_type_t".to_owned(), true)));
    assert!(
        prototypes(&common)
            .iter()
            .any(|p| p.contains("dv_infer_sync("))
    );
    let facts = layout_facts(&crate_file("src/v3_0.rs"));
    assert!(facts.contains(&("Offset of field: dv_model::hw_cycles".to_owned(), 40)));
}

#[test]
fn parsers_account_for_everything() {
    let mut layout_total = 0;
    for bindings in std::iter::once(COMMON_BINDINGS).chain(VERSIONS.iter().map(|v| v.bindings)) {
        let text = crate_file(bindings);
        let parsed = layout_facts(&text).len();
        assert_eq!(
            parsed,
            text.matches("usize]").count(),
            "layout facts in {bindings}"
        );
        if bindings == COMMON_BINDINGS || bindings == "src/v3_0.rs" {
            layout_total += parsed;
        }
    }
    assert!(
        layout_total > 200,
        "only {layout_total} layout facts in common + 3.0"
    );

    for header in std::iter::once(COMMON_HEADER).chain(VERSIONS.iter().map(|v| v.header)) {
        let text = crate_file(header);
        assert_eq!(
            enumerators(&text).len(),
            enum_assignment_lines(&text),
            "enumerators in {header}"
        );
        assert_eq!(
            prototypes(&text).len(),
            prototype_lines(&text),
            "prototypes in {header}"
        );
    }
    let common = crate_file(COMMON_HEADER);
    assert!(enumerators(&common).len() > 20 && prototypes(&common).len() > 20);
}
