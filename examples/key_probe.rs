//! Diagnóstico: imprime cada evento de teclado que rdev detecta, con su nombre y
//! (para teclas sin variante nombrada) el código crudo `Unknown(n)`. Corré esto y
//! presioná la tecla PTT para ver exactamente qué evento genera el teclado.
//!
//!   cargo run --example key_probe

fn main() {
    println!("Escuchando teclado global. Presioná la tecla PTT (Ctrl+C para salir)...");
    if let Err(error) = rdev::listen(|event| match event.event_type {
        rdev::EventType::KeyPress(key) => println!("KeyPress: {key:?}"),
        rdev::EventType::KeyRelease(key) => println!("KeyRelease: {key:?}"),
        _ => {}
    }) {
        eprintln!("Error escuchando teclado: {error:?}");
    }
}
