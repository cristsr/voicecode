//! App icon. Included (not `mod`-declared as a normal file) so `build.rs` can
//! share the exact same pixel generation for the packaged .exe resource.

include!("../icon_render.rs");
