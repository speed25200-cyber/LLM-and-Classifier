// Pas de console supplementaire sous Windows en version publiee.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    prophet_studio_lib::run()
}
