// CensorChip – ONNX Runtime downloader
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Automatically downloads and extracts onnxruntime.dll on first launch
// if it's not already present next to the executable.

use anyhow::{anyhow, Result};
use std::io::Read;
use std::path::PathBuf;

/// Detect the Windows architecture and return the GitHub release filename.
fn get_ort_release_filename() -> Result<&'static str> {
    // Target Windows architectures. For simplicity, default to CPU versions.
    // Users with GPU can manually upgrade to GPU variants.
    #[cfg(target_arch = "x86_64")]
    return Ok("onnxruntime-win-x64-1.24.3.zip");

    #[cfg(target_arch = "aarch64")]
    return Ok("onnxruntime-win-arm64-1.24.3.zip");

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    return Err(anyhow!("Unsupported architecture for ONNX Runtime auto-download"));
}

/// Path inside the zip where the DLL is located.
fn get_ort_dll_path(release: &str) -> String {
    // e.g., "onnxruntime-win-x64-1.24.3/lib/onnxruntime.dll"
    let dirname = release.strip_suffix(".zip").unwrap_or(release);
    format!("{}/lib/onnxruntime.dll", dirname)
}

/// Download and extract onnxruntime.dll if it doesn't exist next to the exe.
pub fn ensure_onnxruntime() -> Result<PathBuf> {
    let exe_dir = std::env::current_exe()?
        .parent()
        .ok_or_else(|| anyhow!("Cannot determine executable directory"))?
        .to_path_buf();

    let dll_path = exe_dir.join("onnxruntime.dll");

    // If it already exists, we're done.
    if dll_path.exists() {
        log::info!("onnxruntime.dll found at {}", dll_path.display());
        return Ok(dll_path);
    }

    log::info!("onnxruntime.dll not found; downloading...");

    let release = get_ort_release_filename()?;
    let url = format!(
        "https://github.com/microsoft/onnxruntime/releases/download/v1.24.3/{}",
        release
    );

    log::info!("Downloading from: {}", url);

    // Download the zip file.
    let response = ureq::get(&url)
        .header("User-Agent", "CensorChip/0.1")
        .call()
        .map_err(|e| anyhow!("Failed to download ONNX Runtime: {}", e))?;

    // Extract the body reader from the response.
    let mut reader = response.into_body().into_reader();
    let mut zip_data = Vec::new();
    
    reader
        .read_to_end(&mut zip_data)
        .map_err(|e| anyhow!("Failed to read download stream: {}", e))?;

    log::info!("Downloaded {} bytes", zip_data.len());

    // Extract the DLL from the zip.
    let reader = std::io::Cursor::new(&zip_data);
    let mut archive = zip::ZipArchive::new(reader)
        .map_err(|e| anyhow!("Failed to parse zip: {}", e))?;

    let dll_inner_path = get_ort_dll_path(release);

    let mut file = archive
        .by_name(&dll_inner_path)
        .map_err(|e| anyhow!("DLL not found in zip at {}: {}", dll_inner_path, e))?;

    // Write to disk.
    let mut dll_bytes = Vec::new();
    file.read_to_end(&mut dll_bytes)
        .map_err(|e| anyhow!("Failed to read DLL from zip: {}", e))?;

    std::fs::write(&dll_path, &dll_bytes)
        .map_err(|e| anyhow!("Failed to write onnxruntime.dll: {}", e))?;

    log::info!(
        "Successfully extracted onnxruntime.dll to {}",
        dll_path.display()
    );

    Ok(dll_path)
}
