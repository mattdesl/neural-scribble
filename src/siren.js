import Random from "canvas-sketch-util/random.js";

export function makeSineLayer(
  inDim,
  outDim,
  omega0 = 1.0,
  isFinal = false,
  jitter = 0,
  layerScale = 1,
) {
  const W = new Float32Array(inDim * outDim);
  const b = new Float32Array(outDim);

  function initialize(isFirst = false) {
    const scale = isFirst ? 1.0 / inDim : Math.sqrt(6.0 / inDim) / omega0;
    for (let i = 0; i < W.length; i++) {
      W[i] = (Random.value() * 2 - 1) * scale;
    }
    b.fill(0);
  }

  function forward(x, out) {
    let ptrW = 0;
    for (let j = 0; j < outDim; j++) {
      let acc = b[j];
      for (let i = 0; i < inDim; i++, ptrW++) {
        acc += W[ptrW] * x[i];
      }
      const jf = jitter === 0 ? 0 : Random.gaussian(0, jitter);
      out[j] = isFinal
        ? acc * layerScale
        : Math.sin(omega0 * acc * layerScale + jf);
    }
  }

  return {
    W,
    b,
    inDim,
    outDim,
    omega0,
    initialize,
    forward,
  };
}

export function SirenNetwork(opts = {}) {
  const {
    inputs = 2,
    outputs = 1,
    perLayerJitter = 0,
    scale: w0 = 30,
    layerScales = [],
  } = opts;
  let { hiddenLayers: hiddenLayersRaw = [] } = opts;

  if (!Array.isArray(hiddenLayersRaw)) {
    hiddenLayersRaw = [hiddenLayersRaw];
  }

  const hiddenLayers = hiddenLayersRaw.map((h) =>
    typeof h === "number" ? h : (h.size ?? 32),
  );

  const dims = [inputs, ...hiddenLayers, outputs];
  const layers = [];
  let totalSize = 0;
  const maxDim = Math.max(...dims);

  for (let i = 0; i < dims.length - 1; i++) {
    const inD = dims[i];
    const outD = dims[i + 1];
    const isFirst = i === 0;
    const isFinal = i === dims.length - 2;
    const omega0 = isFirst ? w0 : 1;
    const layer = makeSineLayer(
      inD,
      outD,
      omega0,
      isFinal,
      isFirst ? 0 : perLayerJitter,
      layerScales[i - 1] || 1,
    );
    layers.push(layer);
    totalSize += layer.W.length + layer.b.length;
  }

  const bufA = new Float32Array(maxDim);
  const bufB = new Float32Array(maxDim);

  function initialize() {
    layers.forEach((L, idx) => L.initialize(idx === 0));
  }

  function toFlat(vec = new Float32Array(totalSize)) {
    let p = 0;
    for (const L of layers) {
      vec.set(L.W, p);
      p += L.W.length;
      vec.set(L.b, p);
      p += L.b.length;
    }
    return vec;
  }

  function fromFlat(vec) {
    let p = 0;
    for (const L of layers) {
      L.W.set(vec.subarray(p, p + L.W.length));
      p += L.W.length;
      L.b.set(vec.subarray(p, p + L.b.length));
      p += L.b.length;
    }
  }

  function forward(input, output = new Float32Array(outputs)) {
    let cur = input;
    let nxt = bufA;

    for (let i = 0; i < layers.length; i++) {
      const target = i === layers.length - 1 ? output : nxt;
      layers[i].forward(cur, target);
      if (i < layers.length - 1) {
        cur = nxt;
        nxt = nxt === bufA ? bufB : bufA;
      }
    }

    return output;
  }

  initialize();

  return {
    type: "siren",
    toFlat,
    fromFlat,
    forward,
  };
}
