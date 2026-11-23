/* Promise-based Pyodide Web Worker RPC client with monotonic run_id cancellation & crash recovery */

export class DagClient {
  constructor(workerPath, onBootStatus, onLog) {
    this.worker = new Worker(workerPath);
    this.nextMsgId = 1;
    this.runId = 0;
    this.pending = new Map();
    this.onBootStatus = onBootStatus || (() => {});
    this.onLog = onLog || (() => {});
    this.readyState = "booting";
    this.bootError = null;
    this._initPromise = null;

    this.worker.onmessage = (event) => {
      const msg = event.data;
      if (msg.type === "boot_status") {
        if (msg.status === "ready") {
          this.readyState = "ready";
        } else if (msg.status === "error") {
          this.readyState = "error";
        }
        this.onBootStatus(msg.status, msg.message);
        this.onLog(`[WASM] ${msg.message}`);
        return;
      }
      const resolver = this.pending.get(msg.id);
      if (!resolver) return;
      this.pending.delete(msg.id);
      if (msg.ok) {
        resolver.resolve(msg.result);
      } else {
        resolver.reject(new Error(msg.error || "Unknown worker error"));
      }
    };

    this.worker.onerror = (err) => {
      const msg = err?.message || "Worker crash";
      this.readyState = "error";
      this.bootError = new Error(msg);
      this.onBootStatus("error", msg);
      this.onLog(`[WASM ERROR] ${msg}`);
      for (const [, resolver] of this.pending.entries()) {
        resolver.reject(new Error(`WASM Worker hiba: ${msg}`));
      }
      this.pending.clear();
    };
  }

  nextRunId() {
    this.runId += 1;
    return this.runId;
  }

  init() {
    if (this._initPromise) return this._initPromise;
    this._initPromise = new Promise((resolve, reject) => {
      const id = this.nextMsgId++;
      this.pending.set(id, {
        resolve: (v) => {
          this.readyState = "ready";
          resolve(v);
        },
        reject: (err) => {
          this.readyState = "error";
          this.bootError = err;
          reject(err);
        },
      });
      this.worker.postMessage({ id, type: "init" });
    });
    return this._initPromise;
  }

  async waitUntilReady() {
    if (this.readyState === "ready") return true;
    if (this.readyState === "error") {
      throw this.bootError || new Error("A Pyodide WASM kernel indítása sikertelen.");
    }
    if (this._initPromise) {
      await this._initPromise;
      return true;
    }
    await this.init();
    return true;
  }

  async call(method, payload) {
    if (this.readyState === "error") {
      throw this.bootError || new Error("A Pyodide WASM motor nem érhető el.");
    }
    return new Promise((resolve, reject) => {
      const id = this.nextMsgId++;
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, type: "rpc", method, payload });
    });
  }
}
