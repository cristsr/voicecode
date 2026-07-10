// Pure pixel generation for the app icon, with no crate dependencies so this
// file can be shared, via `include!`, between the library (runtime tray icon)
// and `build.rs` (packaged .exe icon resource) without a circular dependency.

/// Eases `x` from 0 to 1 as it crosses from `edge0` to `edge1`, for antialiasing.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Signed distance from `(px, py)` to a capsule (a thick line segment) between
/// `(ax, ay)` and `(bx, by)` with the given radius.
fn capsule_sdf(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32, r: f32) -> f32 {
    let (pax, pay) = (px - ax, py - ay);
    let (bax, bay) = (bx - ax, by - ay);
    let h = ((pax * bax + pay * bay) / (bax * bax + bay * bay)).clamp(0.0, 1.0);
    let (dx, dy) = (pax - bax * h, pay - bay * h);
    (dx * dx + dy * dy).sqrt() - r
}

/// Renders the app icon at `size`x`size` as an indigo circular badge with a
/// white microphone glyph, returning tightly packed RGBA8 pixels (no image
/// asset needed).
pub fn render_rgba(size: u32) -> Vec<u8> {
    const BG: (f32, f32, f32) = (79.0, 70.0, 229.0);
    const FG: (f32, f32, f32) = (255.0, 255.0, 255.0);

    let size_f = size as f32;
    let center = size_f / 2.0;
    let bg_radius = size_f * 0.47;

    // Microphone glyph: a capsule head, a "U"-shaped stand ring below it, and
    // a leg + base connecting it to the bottom of the badge.
    let body_top = center - size_f * 0.24;
    let body_bottom = center - size_f * 0.02;
    let body_radius = size_f * 0.10;

    let stand_center_y = body_bottom - size_f * 0.01;
    let stand_radius = size_f * 0.145;
    let stand_thickness = size_f * 0.035;

    let leg_top = stand_center_y + stand_radius - size_f * 0.01;
    let leg_bottom = center + size_f * 0.26;
    let leg_radius = size_f * 0.02;

    let base_half_width = size_f * 0.11;
    let base_radius = size_f * 0.02;

    let mut rgba = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;

            let dist_bg = ((px - center).powi(2) + (py - center).powi(2)).sqrt() - bg_radius;
            let alpha_bg = 1.0 - smoothstep(-1.0, 1.0, dist_bg);

            let body_d = capsule_sdf(px, py, center, body_top, center, body_bottom, body_radius);
            let stand_d = {
                let (dx, dy) = (px - center, py - stand_center_y);
                let ring = (dx * dx + dy * dy).sqrt() - stand_radius;
                // Only the lower half renders, leaving the ring open at the top.
                if dy >= 0.0 {
                    ring.abs() - stand_thickness / 2.0
                } else {
                    f32::MAX
                }
            };
            let leg_d = capsule_sdf(px, py, center, leg_top, center, leg_bottom, leg_radius);
            let base_d = capsule_sdf(
                px,
                py,
                center - base_half_width,
                leg_bottom,
                center + base_half_width,
                leg_bottom,
                base_radius,
            );

            let glyph_d = body_d.min(stand_d).min(leg_d).min(base_d);
            let alpha_glyph = 1.0 - smoothstep(-1.0, 1.0, glyph_d);

            let lerp = |a: f32, b: f32| a + (b - a) * alpha_glyph;
            rgba.push(lerp(BG.0, FG.0).round() as u8);
            rgba.push(lerp(BG.1, FG.1).round() as u8);
            rgba.push(lerp(BG.2, FG.2).round() as u8);
            rgba.push((alpha_bg * 255.0).round() as u8);
        }
    }
    rgba
}
