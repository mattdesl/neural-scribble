import canvasSketch from "canvas-sketch";
import {
  clamp,
  inverseLerp,
  lerp,
  clamp01,
  lerpFrames,
  mod,
  dampArray,
  degToRad,
} from "canvas-sketch-util/math";
import * as Color from "@texel/color";
import * as Random from "canvas-sketch-util/random.js";
import sNES from "./snes.js";
import { createBlitter, createConnection } from "./connection.js";
import { SirenNetwork } from "./siren.js";
import { vec3 } from "gl-matrix";
import mixbox from "mixbox";

//614850
let seed = "" || Random.getRandomSeed();
Random.setSeed(seed);
console.log("Seed:", Random.getSeed());

const vocab = Random.pick(["a blue jay"]);
console.log(vocab);

// Sketch settings
const settings = {
  suffix: [Random.getSeed(), vocab].join("-"),
  dimensions: [128, 128],
};

const sketch = async (props) => {
  const { width, height, context } = props;

  const transformColorSpace = Color.OKLab;
  const colorGamut =
    context.getContextAttributes().colorSpace === "display-p3"
      ? Color.DisplayP3Gamut
      : Color.sRGBGamut;
  const colorSpace = colorGamut.space;
  const useMixbox = true;
  const globalAngle = degToRad(0); //Random.range(-1, 1) * Math.PI * 1;
  const maxNeighbors = Infinity;
  const mode = "manhattan";

  const TARGET_SIZE = 24;
  const TOTAL_EPOCHS = 500;
  const POPULATION_COUNT = 20;
  const globalSigma = 0.2;
  const nodeCount = 5;

  let epochs = 0;

  const connection = await createConnection();
  connection.sendPrompt(vocab);
  window.setPrompt = (prompt) => connection.sendPrompt(prompt);

  const background = "#fff";
  const fitnessBackground = background;
  const blitter = createBlitter({
    size: TARGET_SIZE,
    background,
    draw: drawFitness,
  });

  const tmpCanvas = document.createElement("canvas");
  const tmpContext = tmpCanvas.getContext("2d", { willReadFrequently: true });

  const defaultPositions = Array(nodeCount)
    .fill()
    .map(() => {
      return {
        position: [Random.gaussian(), Random.gaussian()],
      };
    });
  const colorSigma = 1;
  const nodeParams = [
    {
      type: "oklab",
      dimensions: 3,
      sigma: [colorSigma * 1, colorSigma * 1, colorSigma * 1],
      initScale: 0,
    },
    {
      type: "position",
      dimensions: 2,
      sigma: 1,
      initScale: 1,
    },
    // { type: "temperature", dimensions: 1, sigma: 1, initScale: 0 },
  ].filter(Boolean);

  const paramsPerNode = nodeParams.reduce((sum, a) => sum + a.dimensions, 0);
  const solutionLength = nodeCount * paramsPerNode;

  const optimizer = sNES({
    alpha: 1,
    mirrored: true,
    solutionLength,
    populationCount: POPULATION_COUNT,
    random: Random.value,
  });

  const initialData = {
    nodes: Array(nodeCount)
      .fill()
      .map((_, nodeIndex) => {
        const obj = {};
        for (let params of nodeParams) {
          const { type, dimensions, initData, initScale = 1 } = params;
          if (typeof initData === "function") {
            obj[type] = initData(params, nodeIndex);
          } else {
            obj[type] = Array(dimensions)
              .fill()
              .map(
                (_, i) =>
                  Random.gaussian() *
                  (Array.isArray(initScale) ? initScale[i] : initScale),
              );
          }
        }
        return obj;
      }),
  };

  optimizer.center.set(flatten(initialData));

  let idx = 0;
  const sigmaArray = optimizer.sigma;
  for (let i = 0; i < nodeCount; i++) {
    for (let { dimensions, sigma = 1 } of nodeParams) {
      for (let c = 0; c < dimensions; c++) {
        let s = Array.isArray(sigma) ? sigma[c] : sigma;
        sigmaArray[idx++] = s * globalSigma;
      }
    }
  }

  document.body.appendChild(blitter.canvas);
  updateLoop();

  const render = (props) => {
    const { context, width, height } = props;
    drawSolution({
      ...props,
      exporting: props.exporting,
      background,
      fitness: false,
      solution: optimizer.center,
    });
  };

  return { render };

  function sampleDirection(opts) {
    const { sites } = opts;

    const weights = sampleDirectionWeights(opts);
    // let weights = new Array(sites.length).fill(0);
    // const blur = Math.min(width, height) * 0.02;
    // for (let i = 0; i < iterations; i++) {
    //   const nxOff = Random.gaussian(nx, offset * blur * 0);
    //   const nyOff = Random.gaussian(ny, offset * blur * 0);
    //   const w = sampleDirectionWeights({
    //     ...opts,
    //     nx: nxOff,
    //     ny: nyOff,
    //   });
    //   for (let j = 0; j < w.length; j++) {
    //     weights[j] += w[j];
    //   }
    // }
    // for (let i = 0; i < weights.length; i++) {
    //   weights[i] /= iterations;
    // }

    if (useMixbox) {
      const latent = new Array(mixbox.LATENT_SIZE).fill(0);
      for (let j = 0; j < sites.length; j++) {
        const w = weights[j];
        if (w === 0) continue;
        const rgb = Color.convert(
          sites[j].color,
          transformColorSpace,
          Color.sRGB,
        );
        const curLatent = mixbox.floatRgbToLatent(rgb[0], rgb[1], rgb[2]);
        for (let c = 0; c < latent.length; c++) {
          latent[c] += curLatent[c] * w;
        }
      }
      return mixbox.latentToFloatRgb(latent);
    } else {
      const oklab = [0, 0, 0];
      for (let j = 0; j < sites.length; j++) {
        const w = weights[j];
        if (w === 0) continue;
        const color = sites[j].color;
        oklab[0] += color[0] * w;
        oklab[1] += color[1] * w;
        oklab[2] += color[2] * w;
      }
      const oklch = Color.convert(oklab, transformColorSpace, Color.OKLCH);
      const mapped = Color.gamutMapOKLCH(
        oklch,
        colorGamut,
        colorSpace,
        undefined,
        Color.MapToL,
      );
      return mapped;
    }
  }

  function sampleDirectionWeights({
    sites,
    nx,
    ny,
    mode = "manhattan",
    maxNeighbors = Infinity,
  }) {
    const weights = computeWeights({ sites, nx, ny, mode, maxNeighbors });
    return weights;
  }

  function computeWeights({
    sites,
    nx,
    ny,
    mode = "manhattan",
    maxNeighbors = Infinity,
  }) {
    let maxLogit = -Infinity;

    // nx += fractalNoise(Random.noise2D, globalOffset, ny, 1, 12) * 0.02;

    // const angle = Random.gaussian(0, Math.PI * 2);
    // const off = 0.02;
    // nx += off * Math.cos(angle);
    // ny += off * Math.sin(angle);

    const globalCos = Math.cos(globalAngle);
    const globalSin = Math.sin(globalAngle);
    const logits = sites.map((site) => {
      let dx = nx - site.position[0];
      let dy = ny - site.position[1];

      // global rotation
      const gx = globalCos * dx + globalSin * dy;
      const gy = -globalSin * dx + globalCos * dy;

      // per-site rotation
      const cos = Math.cos(site.angle);
      const sin = Math.sin(site.angle);
      const lx = cos * gx + sin * gy;
      const ly = -sin * gx + cos * gy;

      const sx = lx * site.stretch;
      const sy = ly;

      let dist;
      if (mode === "euclidean") {
        dist = Math.hypot(sx, sy);
      } else if (mode === "chebyshev") {
        dist = Math.max(Math.abs(sx), Math.abs(sy));
      } else {
        dist = Math.abs(sx) + Math.abs(sy);
      }

      // dist += Random.gaussian(0, 0.1);
      // dist += fractalNoise3D(Random.noise3D, nx, ny, site.index * 1, 0.01, 8);
      // dist += Random.noise3D(nx, ny, site.index * 1, 4, 0.005);

      const logit = -(dist - site.radius);
      if (logit > maxLogit) maxLogit = logit;
      return logit;
    });

    let totalWeight = 0;
    const weights = logits.map((logit, j) => {
      const w = Math.exp(sites[j].temperature * (logit - maxLogit));
      totalWeight += w;
      return w;
    });

    for (let j = 0; j < weights.length; j++) weights[j] /= totalWeight;

    const K = Math.min(maxNeighbors ?? Infinity, weights.length);
    if (Number.isFinite(K) && K < weights.length) {
      const idx = weights.map((w, i) => [w, i]).sort((a, b) => b[0] - a[0]);
      const keep = new Set(idx.slice(0, K).map(([, i]) => i));
      let sumKept = 0;
      for (let i = 0; i < weights.length; i++) {
        if (!keep.has(i)) {
          weights[i] = 0;
        } else {
          sumKept += weights[i];
        }
      }
      if (sumKept > 0) {
        for (let i = 0; i < weights.length; i++) weights[i] /= sumKept;
      }
    }

    return weights;
  }

  function extractColor(node, { outputOKLab } = {}) {
    let style = "black";
    let curOK;
    if (node.oklab || node.srgb) {
      let mappedColorRGB;
      if (node.oklab) {
        const learned = Color.OKLab;
        const inOKLab = activate(node.oklab, learned);
        Color.convert(inOKLab, learned, Color.OKLab, inOKLab);
        let avgOKLab = inOKLab;
        if (node.srgb) {
          const inRGBToOKLab = Color.convert(
            activate(node.srgb, Color.sRGB),
            Color.sRGB,
            Color.OKLab,
          );
          const mix = clamp01(sigmoid(node.colorMix[0]));
          avgOKLab = vec3.lerp([], inOKLab, inRGBToOKLab, mix);
        }
        mappedColorRGB = gamutMapFromLearned(avgOKLab, Color.OKLab, Color.sRGB);
      } else {
        mappedColorRGB = activate(node.srgb, Color.sRGB);
      }
      curOK = Color.convert(mappedColorRGB, Color.sRGB, Color.OKLab);
      style = Color.serialize(mappedColorRGB, Color.sRGB);
    }
    if (outputOKLab) {
      if (curOK) return curOK.slice();
      else return [0, 0, 0];
    }
    return style;
  }

  function drawSolution(opts = {}) {
    const {
      clear = true,
      fitness = false,
      background = "white",
      context,
      width,
      height,
      exporting,
      // context: outContext,
      // width: outWidth,
      // height: outHeight,
      solution,
    } = opts;

    const learned = unflatten(solution);
    const { nodes } = learned;

    if (clear) {
      context.clearRect(0, 0, width, height);
      context.fillStyle = background;
      context.fillRect(0, 0, width, height);
    }

    context.fillStyle = "black";
    context.strokeStyle = "black";

    const lineWidthScale = fitness ? 2 : 1;
    const lineWidth = 0.01 * width * lineWidthScale;
    const lineJoin = "round";
    const lineCap = "round";
    context.lineWidth = lineWidth;
    context.lineJoin = lineJoin;
    context.lineCap = lineCap;

    const dim = Math.min(width, height);
    const padding = 0.1 * dim;

    // const useMixbox = false;
    // const outputOKLab = !useMixbox;
    const outputOKLab = false;

    const sites = nodes.map((node, i) => {
      const oklab = extractColor(node, { outputOKLab: true });
      const tanhScale = 1;
      const outScale = 0.5;

      const posArray = node.position || defaultPositions[i].position;
      let x = 0.5 + outScale * Math.tanh(tanhScale * posArray[0]);
      let y = 0.5 + outScale * Math.tanh(tanhScale * posArray[1]);
      return {
        color: oklab,
        position: [x, y],
        index: i,
        radius: 0,
        temperature: node.temperature ? sigmoid(node.temperature[0]) * 30 : 10,
        angle: 0,
        stretch: 1,
      };
    });

    tmpCanvas.width = width;
    tmpCanvas.height = height;
    const imageData = tmpContext.createImageData(width, height);
    const data = imageData.data;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const nx = x / width;
        const ny = y / height;

        const rgb = sampleDirection({
          sites,
          nx,
          ny,
          mode,
          maxNeighbors,
        });

        const idx = (y * width + x) * 4;
        data[idx] = Color.floatToByte(rgb[0]);
        data[idx + 1] = Color.floatToByte(rgb[1]);
        data[idx + 2] = Color.floatToByte(rgb[2]);
        data[idx + 3] = 255;
      }
    }
    tmpContext.putImageData(imageData, 0, 0);
    context.drawImage(tmpCanvas, 0, 0, width, height);

    if (!fitness && !exporting) {
      for (let node of sites) {
        const color = Color.serialize(node.color, Color.OKLab, Color.sRGB);
        context.fillStyle = color;
        context.beginPath();
        const px = node.position[0] * width;
        const py = node.position[1] * height;
        const rad = 0.015 * width;
        context.arc(px, py, rad, 0, Math.PI * 2);
        context.fill();
        context.lineWidth = width * 0.0035;
        context.strokeStyle = "white";
        context.stroke();
      }
    }
  }

  function activateHue(angle) {
    // reduce the angle
    angle = angle % 360;

    // force it to be the positive remainder, so that 0 <= angle < 360
    angle = (angle + 360) % 360;

    return angle;
  }

  function gamutMapFromLearned(
    input,
    learnedColorSpace = Color.OKLab,
    outputSpace = colorGamut.space,
  ) {
    return Color.gamutMapOKLCH(
      Color.convert(input, learnedColorSpace, Color.OKLCH),
      colorGamut,
      outputSpace,
      undefined,
      Color.MapToAdaptiveCuspL,
    );
  }

  function activate(color, learnedColorSpace = Color.OKLab) {
    let [x, y, z] = color;
    const abScale = 0.4;
    if (learnedColorSpace == Color.sRGB) {
      return color.map((n) => clamp01(sigmoid(n)));
    } else if (learnedColorSpace == Color.OKLab) {
      return [
        clamp01(sigmoid(color[0])),
        clamp(Math.tanh(color[1]) * abScale, -abScale, abScale),
        clamp(Math.tanh(color[2]) * abScale, -abScale, abScale),
      ];
    } else if (learnedColorSpace == Color.OKHSL) {
      return [activateHue(color[0]), sigmoid(color[1]), sigmoid(color[2])];
    } else if (learnedColorSpace == Color.OKLCH) {
      return [
        sigmoid(color[0]),
        sigmoid(color[1]) * abScale,
        activateHue(color[2]),
      ];
    }
    return [x, y, z];
  }

  function drawFitness(opts) {
    drawSolution({
      ...opts,
      exporting: false,
      background: fitnessBackground,
      fitness: true,
    });
  }

  function flatten(data) {
    const out = new Float32Array(solutionLength);
    let idx = 0;
    for (let obj of data.nodes) {
      for (let { type, dimensions } of nodeParams) {
        for (let c = 0; c < dimensions; c++) {
          const v = obj[type][c];
          out[idx++] = v;
        }
      }
    }
    return out;
  }

  function unflatten(solution, opts = {}) {
    const data = {
      nodes: [],
    };
    let idx = 0;
    for (let i = 0; i < nodeCount; i++) {
      const node = {};
      for (let { type, dimensions } of nodeParams) {
        const shouldKeep = !opts.only || opts.only.includes(type);
        if (shouldKeep) {
          const arr = Array(dimensions).fill(0);
          for (let c = 0; c < dimensions; c++) {
            arr[c] = solution[idx++];
          }
          node[type] = arr;
        } else {
          idx += dimensions;
        }
      }
      data.nodes.push(node);
    }
    return data;
  }

  async function updateLoop() {
    const { bestFitness } = await blitter.tick({
      connection,
      optimizer,
    });
    epochs++;
    console.log(
      `Epoch ${epochs} / ${TOTAL_EPOCHS} - ${bestFitness.toFixed(4)}`,
    );

    props.render();
    if (epochs >= TOTAL_EPOCHS) {
      console.log("Done");
    }
    if (epochs < TOTAL_EPOCHS) {
      requestAnimationFrame(updateLoop);
    }
  }
};

canvasSketch(sketch, settings);

export function vectorToKey(vector, precision = 4) {
  return vector.map((v) => round(v, precision)).join(",");
}

export function round(value, decimals) {
  return Number(Math.round(value + "e" + decimals) + "e-" + decimals);
}

export function drawGraph({ positions, edges, context, color = "black" } = {}) {
  context.beginPath();
  for (let edge of edges) {
    const [start, end] = edge.map((i) => positions[i]);
    context.moveTo(start[0], start[1]);
    context.lineTo(end[0], end[1]);
  }
  context.strokeStyle = color;
  context.stroke();
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function drawOffsetPath(context, offsetPathResult, closed = false) {
  context.beginPath();
  if (closed) {
    offsetPathResult.edges.forEach((edge, i) => {
      const rev = i > 0;
      if (rev) {
        edge = edge.slice();
        edge.reverse();
      }
      for (let j = 0; j < edge.length; j++) {
        const p = edge[j];
        if (j === 0) context.moveTo(p[0], p[1]);
        else context.lineTo(p[0], p[1]);
      }
      context.closePath();
    });
  } else {
    const vertices = offsetPathResult.vertices;
    for (let i = 0; i < vertices.length; i++) {
      const point = vertices[i];
      context.lineTo(point[0], point[1]);
    }
    context.closePath();
  }
  context.fill();
}

function ema(series, alpha = 0.2) {
  if (!series.length) return [];
  const out = new Array(series.length);
  out[0] = series[0];
  for (let i = 1; i < series.length; i++) {
    out[i] = alpha * series[i] + (1 - alpha) * out[i - 1];
  }
  return out;
}

function drawPlot(context, plot, bounds, width, height) {
  for (let i = 0; i < plot.length; i++) {
    const t = plot.length <= 1 ? 0 : i / (plot.length - 1);
    const f = plot[i];
    const v = clamp01(inverseLerp(bounds[0], bounds[1], f));
    context.lineTo(t * width, (1 - v) * height);
  }
}

function softmax(logits, temperature = 1) {
  const t = Math.max(1e-6, temperature);
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i] / t);

  const exps = new Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] / t - max);
    exps[i] = e;
    sum += e;
  }
  for (let i = 0; i < exps.length; i++) exps[i] /= sum;
  return exps;
}
