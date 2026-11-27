/* Promise-based Pyodide Web Worker RPC client with per-kind stale filtering, progress callbacks & crash recovery */

export class DagClient {
  constructor(workerPath, onBootStatus, onLog, onProgress) {
    this.workerPath = workerPath;
    this.nextMsgId = 1;
    this.runId = 0;
    this.pending = new Map();
    this.latestRunByKind = new Map();
    this.onBootStatus = onBootStatus || (() => {});
    this.onLog = onLog || (() => {});
    this.onProgress = onProgress || (() => {});
    this.readyState = "booting";
    this.bootError = null;
    this._initPromise = null;
    this.worker = null;
    this._spawnWorker();
  }

  _spawnWorker() {
    this.worker = new Worker(this.workerPath);
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
      if (msg.type === "progress") {
        this.onProgress(msg);
        return;
      }
      const resolver = this.pending.get(msg.id);
      if (!resolver) return;
      this.pending.delete(msg.id);
      if (resolver.kind && resolver.runId != null) {
        const latest = this.latestRunByKind.get(resolver.kind);
        if (latest != null && resolver.runId < latest) {
          const cancelledErr = new Error(`Superseded by newer ${resolver.kind} request (${latest})`);
          cancelledErr.name = "CancelledError";
          resolver.reject(cancelledErr);
          return;
        }
      }
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
        kind: "init",
        runId: null,
        resolve: (v) => {
          this.readyState = "ready";
          resolve(v);
        },
        reject: (err) => {
          this.readyState = "error";
          this.bootError = err;
          this._initPromise = null;
          reject(err);
        },
      });
      this.worker.postMessage({ id, type: "init" });
    });
    return this._initPromise;
  }

  async restart() {
    if (this.worker) {
      try {
        this.worker.terminate();
      } catch (_) {}
    }
    for (const [, resolver] of this.pending.entries()) {
      const err = new Error("WASM Worker újraindítva.");
      err.name = "CancelledError";
      resolver.reject(err);
    }
    this.pending.clear();
    this.latestRunByKind.clear();
    this.bootError = null;
    this.readyState = "booting";
    this._initPromise = null;
    this._spawnWorker();
    this.onBootStatus("booting", "WASM kernel újraindítása...");
    return this.init();
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
    const kind = String(method || "rpc");
    let reqRunId = null;
    if (payload && typeof payload === "object" && payload.run_id != null && Number.isFinite(Number(payload.run_id))) {
      reqRunId = Number(payload.run_id);
      this.latestRunByKind.set(kind, Math.max(reqRunId, this.latestRunByKind.get(kind) ?? -Infinity));
    }
    return new Promise((resolve, reject) => {
      const id = this.nextMsgId++;
      this.pending.set(id, { kind, runId: reqRunId, resolve, reject });
      this.worker.postMessage({ id, type: "rpc", method, payload });
    });
  }

  runStage6(recipe, runId = this.nextRunId()) {
    return this.call("run_stage_6", { recipe, run_id: runId });
  }
}
