//! Diagnostic tool: prints every keyboard event `rdev` detects, with its name
//! or, for keys without a named variant, the raw `Unknown(n)` code. Run this
//! and press the PTT key to see exactly what event the keyboard generates.
//!
//!   cargo run --example key_probe

fn main() {
    println!("Listening globally. Press the PTT key (Ctrl+C to quit)...");
    if let Err(error) = rdev::listen(|event| match event.event_type {
        rdev::EventType::KeyPress(key) => println!("KeyPress: {key:?}"),
        rdev::EventType::KeyRelease(key) => println!("KeyRelease: {key:?}"),
        _ => {}
    }) {
        eprintln!("Error listening to keyboard: {error:?}");
    }
}
