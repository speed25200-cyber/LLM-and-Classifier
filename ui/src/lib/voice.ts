// Voix cote interface : capture micro (AudioWorklet -> PCM 16 kHz), push-to-talk, mains libres (flux vers le VAD du
// coeur), synthese vocale phrase par phrase (sherpa-onnx cote coeur, repli Web Speech si la voix n'est pas installee).
import { api, ApiError, wsUrl } from "./api";
import { app } from "./store.svelte";
import { runVoiceCommand } from "./commands";
import type { VoiceRoute } from "./types";

const WORKLET = `class Tap extends AudioWorkletProcessor{process(i){const c=i[0]&&i[0][0];if(c)this.port.postMessage(c.slice(0));return true}}registerProcessor('pcm-tap',Tap)`;

function toPcm16(chunks: Float32Array[]): Int16Array {
  const n = chunks.reduce((a, c) => a + c.length, 0);
  const out = new Int16Array(n);
  let o = 0;
  for (const c of chunks) for (let i = 0; i < c.length; i++) out[o++] = Math.max(-1, Math.min(1, c[i])) * 0x7fff;
  return out;
}

class Mic {
  ctx: AudioContext | null = null;
  stream: MediaStream | null = null;
  node: AudioWorkletNode | null = null;
  onChunk: ((c: Float32Array) => void) | null = null;

  async open() {
    if (this.ctx) return;
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    this.ctx = new AudioContext({ sampleRate: 16000 });
    const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
    await this.ctx.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, "pcm-tap");
    this.node.port.onmessage = (m) => {
      const c = m.data as Float32Array;
      let peak = 0;
      for (let i = 0; i < c.length; i += 8) peak = Math.max(peak, Math.abs(c[i]));
      app.voice.level = app.voice.level * 0.6 + Math.min(1, peak * 2.2) * 0.4;
      this.onChunk?.(c);
    };
    src.connect(this.node);
  }

  close() {
    this.node?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    this.ctx?.close();
    this.ctx = null;
    this.stream = null;
    this.node = null;
    app.voice.level = 0;
  }
}

class Speaker {
  private queue: string[] = [];
  private audio: HTMLAudioElement | null = null;
  private playing = false;
  private gen = 0;

  speak(text: string) {
    const parts = text
      .replace(/```[\s\S]*?```/g, " ")
      .split(/(?<=[.!?…])\s+/)
      .map((s) => s.replace(/[*_`#>|]/g, "").trim())
      .filter(Boolean);
    this.queue.push(...parts);
    if (!this.playing) this.next(this.gen);
  }

  stop() {
    this.gen++;
    this.queue = [];
    this.audio?.pause();
    this.audio = null;
    speechSynthesis?.cancel();
    this.playing = false;
    app.voice.speaking = false;
  }

  private async next(gen: number) {
    const s = this.queue.shift();
    if (!s || gen !== this.gen) {
      this.playing = false;
      app.voice.speaking = false;
      return;
    }
    this.playing = true;
    app.voice.speaking = true;
    try {
      const blob = await api<Blob>("/api/voice/tts", { body: { text: s } });
      if (gen !== this.gen) return;
      const url = URL.createObjectURL(blob);
      this.audio = new Audio(url);
      this.audio.onended = () => (URL.revokeObjectURL(url), this.next(gen));
      await this.audio.play();
    } catch (e) {
      if (e instanceof ApiError && e.status === 503 && "speechSynthesis" in window) {
        // voix locale non installee : synthese du systeme (Windows : voix Microsoft hors ligne)
        const u = new SpeechSynthesisUtterance(s);
        u.lang = app.settings?.language === "en" ? "en-US" : "fr-FR";
        u.rate = app.settings?.voice.speed ?? 1;
        u.onend = () => this.next(gen);
        speechSynthesis.speak(u);
      } else {
        this.playing = false;
        app.voice.speaking = false;
      }
    }
  }
}

export const speaker = new Speaker();
const mic = new Mic();
let pttChunks: Float32Array[] = [];
let stream: WebSocket | null = null;

async function handleRoute(route: VoiceRoute | undefined, text: string) {
  app.voice.route = route ?? null;
  if (!route || route.kind === "ignored") return;
  if (route.kind === "command" && route.command) {
    app.toast("voice", `Commande vocale : ${route.command.replace(/_/g, " ")}`, `« ${route.text} » · ${route.source === "systemone" ? `classifieur ${Math.round(route.confidence * 100)} %` : "grammaire"} · ${route.ms} ms`, 2600);
    await runVoiceCommand(route.command);
    return;
  }
  const prompt = route.text || text;
  if (prompt) await app.send(prompt);
}

export async function startPtt() {
  if (app.voice.recording || app.voice.handsfree) return;
  speaker.stop();
  app.voice.error = "";
  try {
    pttChunks = [];
    await mic.open();
    mic.onChunk = (c) => pttChunks.push(c);
    app.voice.recording = true;
    app.voice.transcript = "";
    app.voice.route = null;
  } catch (e) {
    app.voice.error = "Micro indisponible";
    app.toast("error", "Micro indisponible", String(e));
  }
}

export async function stopPtt() {
  if (!app.voice.recording) return;
  app.voice.recording = false;
  mic.onChunk = null;
  mic.close();
  const pcm = toPcm16(pttChunks);
  pttChunks = [];
  if (pcm.length < 16000 * 0.35) return;
  app.voice.busy = true;
  try {
    const r = await api<{ text: string; ms: number; route?: VoiceRoute }>("/api/voice/transcribe?sample_rate=16000", {
      raw: new Blob([pcm.buffer as ArrayBuffer]),
      headers: { "Content-Type": "application/octet-stream" },
    });
    app.voice.transcript = r.text;
    await handleRoute(r.route, r.text);
  } catch (e) {
    const msg = e instanceof ApiError && e.status === 503 ? "Installez la reconnaissance vocale (ecran Modeles > Voix)." : String(e);
    app.toast("warn", "Reconnaissance vocale indisponible", msg, 6000);
  } finally {
    app.voice.busy = false;
  }
}

export async function toggleHandsfree(on?: boolean) {
  const want = on ?? !app.voice.handsfree;
  if (!want) {
    stream?.close();
    stream = null;
    mic.onChunk = null;
    mic.close();
    app.voice.handsfree = false;
    return;
  }
  try {
    await mic.open();
    stream = new WebSocket(wsUrl("/api/voice/stream"));
    stream.binaryType = "arraybuffer";
    let buf: Float32Array[] = [];
    let n = 0;
    mic.onChunk = (c) => {
      buf.push(c);
      n += c.length;
      if (n >= 1600 && stream?.readyState === WebSocket.OPEN) {
        stream.send(toPcm16(buf).buffer as ArrayBuffer);
        buf = [];
        n = 0;
      }
    };
    stream.onmessage = async (m) => {
      const e = JSON.parse(m.data);
      if (e.type === "error") {
        app.toast("warn", "Mains libres indisponible", e.error, 6000);
        toggleHandsfree(false);
      } else if (e.type === "vad") {
        app.voice.recording = e.speaking;
        if (e.speaking) speaker.stop();
      } else if (e.type === "segment" && e.text) {
        app.voice.transcript = e.text;
        await handleRoute(e.route, e.text);
      }
    };
    stream.onclose = () => {
      if (app.voice.handsfree) toggleHandsfree(false);
    };
    app.voice.handsfree = true;
    app.toast("voice", "Mains libres active", `Dites « ${app.settings?.voice.wake_word ?? "Prophet"}, … » pour parler a l'agent.`);
  } catch (e) {
    app.toast("error", "Micro indisponible", String(e));
    mic.close();
  }
}
