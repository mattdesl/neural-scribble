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

//614850
let seed = "" || Random.getRandomSeed();
Random.setSeed(seed);
console.log("Seed:", Random.getSeed());

const vocab = Random.pick(["a sailboat"]);
console.log(vocab);

// Sketch settings
const settings = {
  suffix: [Random.getSeed(), vocab].join("-"),
  dimensions: [2048, 2048],
};

const sketch = async (props) => {
  const { width, height } = props;

  const PLOT_FITNESS = true;
  const PLOT_FITNESS_BOUNDS = [0.25, 0.6];
  const PLOT_FITNESS_HISTORY = [];
  const SHOW_PLOT_IN_EXPORT = false;

  const TARGET_SIZE = 32;
  const TOTAL_EPOCHS = 500;
  const POPULATION_COUNT = 20;
  const globalSigma = 1;
  const nodeCount = 3;
  const HIDDEN_LAYER_SIZE = 16;
  const HIDDEN_LAYER_COUNT = 2;
  const NET_LEARNING_RATE = 0.005 * 1;
  const NET_SCALE = 30;
  const GAUSSIAN_INPUT_DIMENSION = 2;

  let epochs = 0;

  const outputGamut = Color.sRGBGamut;

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

  const outputCount = 3;
  const inputCount = GAUSSIAN_INPUT_DIMENSION + 2;

  const createNet = () => {
    return SirenNetwork({
      inputs: inputCount,
      outputs: outputCount,
      scale: NET_SCALE,
      hiddenLayers: Array(HIDDEN_LAYER_COUNT).fill(HIDDEN_LAYER_SIZE),
    });
  };

  const tmpNet = createNet();
  const netLength = tmpNet.toFlat().length;
  const tempFloat3 = new Float32Array(netLength);

  const colorSigma = 0.1;
  const nodeParams = [
    {
      type: "oklab",
      dimensions: 3,
      sigma: [colorSigma * 1, colorSigma * 1, colorSigma * 1],
      initScale: 0,
    },
    // { type: "angleStart", dimensions: 1, sigma: 0.1, initScale: 1 },
    {
      type: "net",
      dimensions: netLength,
      sigma: NET_LEARNING_RATE,
      initData: () => createNet().toFlat(),
    },
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

  const gaussianInputs = Array(GAUSSIAN_INPUT_DIMENSION)
    .fill()
    .map(() => Random.gaussian());
  const tmpInput = new Float32Array(inputCount);
  const tmpOutput = new Float32Array(outputCount);

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

    const showPlot = PLOT_FITNESS && (SHOW_PLOT_IN_EXPORT || !props.exporting);
    if (showPlot) {
      context.beginPath();
      drawPlot(
        context,
        PLOT_FITNESS_HISTORY,
        PLOT_FITNESS_BOUNDS,
        width,
        height,
      );
      context.lineWidth = 0.005 * Math.min(width, height);
      context.strokeStyle = "black";
      context.lineJoin = "round";
      context.globalAlpha = 0.25;
      context.stroke();

      const smooth = ema(PLOT_FITNESS_HISTORY, 0.1);
      context.beginPath();
      drawPlot(context, smooth, PLOT_FITNESS_BOUNDS, width, height);
      context.strokeStyle = "green";
      context.globalAlpha = 0.75;
      context.stroke();
      context.globalAlpha = 1;
    }
  };

  return { render };

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

    for (let node of nodes) {
      const color = extractColor(node);
      context.fillStyle = color;
      const vertexCount = fitness ? 100 : 1024;
      const angleLength = Math.PI * 2;
      const vertices = [];
      for (let i = 0; i < vertexCount; i++) {
        const t = i / vertexCount;
        const angleStart = node.angleStart ? node.angleStart[0] : 0;
        let angle = angleStart + t * angleLength;

        if (!fitness && exporting) angle += Random.gaussian(0, 0.02);

        let u = Math.cos(angle);
        let v = Math.sin(angle);

        const amp = 0.0;
        const freq = 10;
        u = u + (amp != 0 ? Random.noise2D(u, v, freq, 0.05) * amp : 0);
        v = v + (amp != 0 ? Random.noise2D(u, v, freq, 0.05) * amp : 0);

        let idx = 0;
        tmpInput[idx++] = u;
        tmpInput[idx++] = v;
        tmpInput.set(gaussianInputs, idx);

        tempFloat3.set(node.net);
        tmpNet.fromFlat(tempFloat3);
        tmpNet.forward(tmpInput, tmpOutput);
        const tanhScale = 0.5;
        tmpOutput[0] = Math.tanh(tanhScale * tmpOutput[0]);
        tmpOutput[1] = Math.tanh(tanhScale * tmpOutput[1]);
        const scl = 1.2;
        const outU = tmpOutput[0] * scl;
        const outV = tmpOutput[1] * scl;
        let x = width / 2 + (outU * (width - padding * 2)) / 2;
        let y = height / 2 + (outV * (height - padding * 2)) / 2;

        vertices.push([x, y]);
      }

      const closed = angleLength >= Math.PI * 2;
      context.beginPath();
      for (let [x, y] of vertices) {
        context.lineTo(x, y);
      }
      if (closed) context.closePath();
      context.stroke();
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
    outputSpace = outputGamut.space,
  ) {
    return Color.gamutMapOKLCH(
      Color.convert(input, learnedColorSpace, Color.OKLCH),
      outputGamut,
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
    PLOT_FITNESS_HISTORY.push(bestFitness);

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
