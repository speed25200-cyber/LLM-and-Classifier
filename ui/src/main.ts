import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import "./styles/app.css";
import { mount } from "svelte";
import App from "./App.svelte";

const target = document.getElementById("app")!;
target.replaceChildren();
mount(App, { target });
