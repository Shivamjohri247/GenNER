//! I/O utilities

use std::path::Path;

/// Get file size in bytes
pub fn file_size(path: impl AsRef<Path>) -> std::io::Result<u64> {
    Ok(std::fs::metadata(path.as_ref())?.len())
}

/// Format bytes as human readable string
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];

    let mut size = bytes as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    if unit_idx == 0 {
        format!("{} {}", bytes, UNITS[unit_idx])
    } else {
        format!("{:.1} {}", size, UNITS[unit_idx])
    }
}

/// Create directory if it doesn't exist
pub fn ensure_dir(path: impl AsRef<Path>) -> std::io::Result<()> {
    std::fs::create_dir_all(path.as_ref())
}

/// Get the extension of a file path
pub fn get_extension(path: impl AsRef<Path>) -> Option<String> {
    path.as_ref()
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase())
}

/// Check if a file exists
pub fn exists(path: impl AsRef<Path>) -> bool {
    path.as_ref().exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(100), "100 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0 GB");
    }

    #[test]
    fn test_get_extension() {
        assert_eq!(get_extension("test.json"), Some("json".to_string()));
        assert_eq!(get_extension("test.JSON"), Some("json".to_string()));
        assert_eq!(get_extension("test"), None);
        assert_eq!(get_extension(".hidden"), None);
    }
}
