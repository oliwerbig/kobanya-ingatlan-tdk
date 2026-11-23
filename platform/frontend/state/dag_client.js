/* Promise-based Pyodide Web Worker RPC client with monotonic run_id cancellation */

export class DagClient {
  constructor(workerPath, onBootStatus, onLog) {
    this.worker = new Worker(workerPath);
    this.nextMsgId = 1;
    this.runId = 0;
    this.pending = new Map();
    this.onBootStatus = onBootStatus || (() => {});
    this.onLog = onLog || (() => {});

    this.worker.onmessage = (event) => {
      const msg = event.data;
      if (msg.type === "boot_status") {
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
      this.onBootStatus("error", err.message || "Worker crash");
      this.onLog(`[WASM ERROR] ${err.message}`);
    };
  }

  nextRunId() {
    this.runId += 1;
    return this.runId;
  }

  init() {
    return new Promise((resolve, reject) => {
      const id = this.nextMsgId++;
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, type: "init" });
    });
  }

  call(method, payload) {
    return new Promise((resolve, reject) => {
      const id = this.nextMsgId++;
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, type: "rpc", method, payload });
    });
  }
}
