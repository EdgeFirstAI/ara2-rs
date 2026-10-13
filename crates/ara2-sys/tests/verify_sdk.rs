//! Checks ara2-sys's DVAPI declarations against a real SDK's headers.
//!
//! For each version whose SDK include directory is named by an environment
//! variable (`ARA2_SDK_V1_1`, `ARA2_SDK_V1_3`, `ARA2_SDK_V3_0`), this
//! builds a C translation unit that includes the SDK's `dvapi.h`, asserts
//! every size, alignment, field offset and enumerator value ara2-sys
//! declares, checks each enum typedef's size and signedness, and then
//! redeclares every function prototype ara2-sys binds, so a mismatched
//! signature is a "conflicting types" compile error. Versions with no
//! variable set are skipped, which is the normal case in CI.

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
/// `("Offset of field: dv_model::num_inputs", 56)`.
fn layout_facts(bindings: &str) -> Vec<(String, u64)> {
    let mut facts = Vec::new();
    let mut rest = bindings;
    while let Some(start) = rest.find("[\"") {
        rest = &rest[start + 2..];
        let Some(end) = rest.find("\"]") else { break };
        let label = rest[..end].to_owned();
        rest = &rest[end + 2..];
        let (Some(minus), Some(close)) = (rest.find(" - "), rest.find("usize]")) else {
            break;
        };
        if minus > close {
            continue;
        }
        let value = rest[minus + 3..close].trim().parse().expect("layout value");
        facts.push((label, value));
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

/// `(name, value)` for every enumerator in one of our headers.
fn enumerators(header: &str) -> Vec<(String, i64)> {
    header
        .lines()
        .filter_map(|line| {
            let line = line.split("/*").next()?.trim().trim_end_matches(',');
            let (name, value) = line.split_once('=')?;
            let name = name.trim();
            let is_ident = !name.is_empty()
                && name
                    .chars()
                    .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');
            if !is_ident {
                return None;
            }
            Some((name.to_owned(), value.trim().parse().ok()?))
        })
        .collect()
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

/// Every one-line function prototype in one of our headers.
fn prototypes(header: &str) -> Vec<String> {
    header
        .lines()
        .map(str::trim)
        .filter(|l| {
            l.ends_with(");") && l.contains("dv_") && !l.starts_with('*') && !l.starts_with("/*")
        })
        .map(str::to_owned)
        .collect()
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
