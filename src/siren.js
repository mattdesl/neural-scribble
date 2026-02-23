import Random from "canvas-sketch-util/random.js";

/**
 * Create a single SIREN layer:
 *  - inDim  = input dimensionality
 *  - outDim = number of output units
 *  - omega0 = ω₀ scaling for the sine (use large value like 30 for first layer, 1 elsewhere)
 *  - isFinal = if true, this layer is “final” (no sine activation, just linear outputs)
 *
 * Usage: layer.forward(x_in, x_out)
 */
export function makeSineLayer(
  inDim,
  outDim,
  omega0 = 1.0,
  isFinal = false,
  jitter = 0,
  layerScale = 1,
) {
  // Flattened weight array of length = inDim * outDim
  const W = new Float32Array(inDim * outDim);
  // Biases, length = outDim
  const b = new Float32Array(outDim);

  /**
   * initialize(isFirst = false):
   *   For the *first* layer, we use ω₀ = w0 (e.g. 30). For others, ω₀ = 1.
   *   Initialization formula from the SIREN paper:
   *     scale = isFirst
   *       ? (1 / inDim)
   *       : (Math.sqrt(6 / inDim) / omega0)
   *   Then W[i] ∼ Uniform(−scale, +scale), b filled with zeros.
   */
  function initialize(isFirst = false) {
    // let scale;
    // if (isFirst) {
    //   scale = 1.0 / inDim;
    // } else if (isFinal) {
    //   // MUCH smaller so outputs don't immediately saturate your outer tanh
    //   scale = 0.1 / Math.sqrt(inDim); // try 0.05..0.2
    // } else {
    //   scale = Math.sqrt(6.0 / inDim) / omega0;
    // }

    // for (let i = 0; i < W.length; i++) {
    //   W[i] = (Random.value() * 2 - 1) * scale;
    // }
    // b.fill(0);

    const scale = isFirst ? 1.0 / inDim : Math.sqrt(6.0 / inDim) / omega0;

    for (let i = 0; i < W.length; i++) {
      // Random.value() returns [0,1), so (2 * value − 1) gives [−1, +1)
      W[i] = (Random.value() * 2 - 1) * scale;
      // W[i] = Math.max(-1, Math.min(1, Random.gaussian(0, scale / 2)));
    }
    // zero out biases
    b.fill(0);
  }

  /**
   * forward(x, out):
   *   x: Float32Array (length = inDim)
   *   out: Float32Array (length = outDim)
   *
   *   Computes: out = ( sine( ω₀ * (W·x + b) ) ) for hidden layers,
   *             or out = (W·x + b) directly if isFinal = true.
   */
  function forward(x, out) {
    let ptrW = 0;
    // For each output neuron j:
    for (let j = 0; j < outDim; j++) {
      // Start with bias
      let acc = b[j];
      // Compute dot(W_row_j, x)
      for (let i = 0; i < inDim; i++, ptrW++) {
        acc += W[ptrW] * x[i];
      }
      const jf = jitter == 0 ? 0 : Random.gaussian(0, jitter);
      // If final layer, no sine; otherwise apply sin(ω₀ * acc)
      out[j] = isFinal
        ? acc * layerScale
        : activate(omega0 * acc * layerScale + jf);
    }
  }

  function activate(x) {
    return Math.sin(x);
  }

  function softTri(x) {
    return (2 / Math.PI) * Math.asin(Math.sin(x));
  }

  function tri(x) {
    // range [-1,1], period 2π
    const t = (x / (2 * Math.PI)) % 1;
    const u = t < 0 ? t + 1 : t; // [0,1)
    const y = 1 - 4 * Math.abs(u - 0.5);
    return y; // [-1,1]
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

/**
 * SirenNetwork(opts):
 *   opts.inputs       = dimensionality of the input (e.g. 2)
 *   opts.outputs      = dimensionality of the output (e.g. 2)
 *   opts.hiddenLayers = array of hidden sizes, e.g. [32, 32]
 *   opts.w0           = ω₀ for the *first* layer (default = 30)
 *
 * Returns an object with:
 *   - initialize(): initialize weights/biases in all layers
 *   - forward(inputArray): returns a Float32Array of length = outputs
 *
 * Internally:
 *   dims = [ inputs, ...hiddenLayers, outputs ]
 *   and it creates one SIREN layer per adjacent pair of dims.
 */
export function SirenNetwork(opts = {}) {
  const {
    inputs = 2,
    outputs = 1,
    perLayerJitter = 0,
    scale: w0 = 30, // default ω₀ for the first SIREN layer,
    layerScales = [],
  } = opts;
  let { hiddenLayers: hiddenLayersRaw = [] } = opts;

  if (!Array.isArray(hiddenLayersRaw)) {
    hiddenLayersRaw = [hiddenLayersRaw];
  }

  // Ensure hiddenLayers is an array of just numbers
  const hiddenLayers = [];
  for (let h of hiddenLayersRaw) {
    if (typeof h == "number") hiddenLayers.push(h);
    else {
      const { size = 32 } = h;
      hiddenLayers.push(size);
    }
  }

  // Build a dims array: [ inputs, ...hiddenLayers, outputs ]
  const dims = [inputs, ...hiddenLayers, outputs];

  // Container for each SIREN layer
  const layers = [];

  let totalSize = 0;

  // Track the maximum dimension (for scratch buffers)
  let maxDim = inputs;
  for (let d of dims) {
    if (d > maxDim) maxDim = d;
  }

  // Create one layer per adjacent pair in dims[]:
  for (let i = 0; i < dims.length - 1; i++) {
    const inD = dims[i];
    const outD = dims[i + 1];
    const isFirst = i === 0;
    const isFinal = i === dims.length - 2;
    // ω₀ = w0 for first layer, otherwise 1.0
    const omega0 = isFirst ? w0 : 1;

    // console.log("LAYER", i, "inD:", inD, "outD:", outD, "omega0:", omega0);
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

  // Two scratch buffers for alternating hidden outputs
  const bufA = new Float32Array(maxDim);
  const bufB = new Float32Array(maxDim);

  /**
   * initialize():
   *  Calls initialize(isFirst) on each layer so that
   *  layer 0 uses w0, subsequent layers use ω₀=1.
   */
  function initialize() {
    layers.forEach((L, idx) => {
      L.initialize(idx === 0);
    });
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

  /**
   * forward(input, output?):
   *   input:  Array or Float32Array of length = inputs
   *   output: (optional) Float32Array of length = outputs
   *           If omitted, a new Float32Array(outputs) is allocated.
   *
   *   Returns: Float32Array of length = outputs
   *
   *   Internally, it alternates between bufA and bufB for hidden layers,
   *   then writes final layer’s results into `output`.
   */
  function forward(input, output = new Float32Array(outputs)) {
    // current “source” = input vector
    let cur = input;
    // next buffer = bufA initially
    let nxt = bufA;

    for (let i = 0; i < layers.length; i++) {
      const L = layers[i];
      // If this is the very last layer, write into `output`
      const target = i === layers.length - 1 ? output : nxt;

      // Compute L.forward(cur, target)
      L.forward(cur, target);

      // For next iteration: swap buffers (unless we’re at final layer)
      if (i < layers.length - 1) {
        cur = nxt;
        nxt = nxt === bufA ? bufB : bufA;
      }
    }

    return output;
  }

  initialize();

  // Expose a similar API to your CPPN:
  return {
    type: "siren",
    toFlat,
    fromFlat,
    forward,
  };
}
