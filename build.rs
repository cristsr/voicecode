// Embeds the app icon as the packaged .exe's resource icon on Windows (the
// icon Explorer/the taskbar show for the file itself). No-op elsewhere.

use std::env;
use std::path::PathBuf;

include!("src/icon_render.rs");

fn main() {
    println!("cargo:rerun-if-changed=src/icon_render.rs");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    let ico_path = write_ico();
    winresource::WindowsResource::new()
        .set_icon(ico_path.to_str().expect("OUT_DIR path is valid UTF-8"))
        .compile()
        .expect("failed to embed the .exe icon resource");
}

/// Renders the app icon at the sizes a Windows .ico commonly bundles and packs
/// them into a single .ico file under `OUT_DIR`.
fn write_ico() -> PathBuf {
    const SIZES: [u32; 5] = [16, 32, 48, 128, 256];

    let mut file = Vec::new();
    file.extend_from_slice(&0u16.to_le_bytes()); // reserved
    file.extend_from_slice(&1u16.to_le_bytes()); // type: icon
    file.extend_from_slice(&(SIZES.len() as u16).to_le_bytes());

    let mut offset = 6 + 16 * SIZES.len() as u32;
    for &size in &SIZES {
        let mask_row_bytes = size.div_ceil(32) * 4;
        let image_size = 40 + size * size * 4 + mask_row_bytes * size;

        // ICONDIRENTRY: 0 means "256" for width/height, per the ICO format.
        file.push(if size == 256 { 0 } else { size as u8 });
        file.push(if size == 256 { 0 } else { size as u8 });
        file.push(0); // color count (0 = no palette)
        file.push(0); // reserved
        file.extend_from_slice(&1u16.to_le_bytes()); // color planes
        file.extend_from_slice(&32u16.to_le_bytes()); // bits per pixel
        file.extend_from_slice(&image_size.to_le_bytes());
        file.extend_from_slice(&offset.to_le_bytes());
        offset += image_size;
    }

    for &size in &SIZES {
        let rgba = render_rgba(size);
        let mask_row_bytes = size.div_ceil(32) * 4;

        // BITMAPINFOHEADER. Per the ICO format, biHeight is doubled to cover
        // the trailing AND mask even though these icons carry a real alpha
        // channel and don't rely on it.
        file.extend_from_slice(&40u32.to_le_bytes());
        file.extend_from_slice(&(size as i32).to_le_bytes());
        file.extend_from_slice(&((size * 2) as i32).to_le_bytes());
        file.extend_from_slice(&1u16.to_le_bytes());
        file.extend_from_slice(&32u16.to_le_bytes());
        file.extend_from_slice(&0u32.to_le_bytes()); // BI_RGB, uncompressed
        file.extend_from_slice(&(size * size * 4 + mask_row_bytes * size).to_le_bytes());
        file.extend_from_slice(&0i32.to_le_bytes());
        file.extend_from_slice(&0i32.to_le_bytes());
        file.extend_from_slice(&0u32.to_le_bytes());
        file.extend_from_slice(&0u32.to_le_bytes());

        // Pixel data: bottom-up rows, BGRA per pixel (Windows bitmap order).
        for y in (0..size).rev() {
            for x in 0..size {
                let i = ((y * size + x) * 4) as usize;
                file.extend_from_slice(&[rgba[i + 2], rgba[i + 1], rgba[i], rgba[i + 3]]);
            }
        }
        // AND mask: all zero (fully opaque); transparency comes from alpha.
        file.extend(std::iter::repeat(0u8).take((mask_row_bytes * size) as usize));
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let ico_path = out_dir.join("app_icon.ico");
    std::fs::write(&ico_path, &file).expect("failed to write the generated .ico");
    ico_path
}
