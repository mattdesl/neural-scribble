import canvasSketch from "canvas-sketch";
import { clamp, inverseLerp, clamp01 } from "canvas-sketch-util/math";
import * as Color from "@texel/color";
import * as Random from "canvas-sketch-util/random.js";
import sNES from "./snes.js";
import { createBlitter, createConnection } from "./connection.js";
import { SirenNetwork } from "./siren.js";

let seed = "" || Random.getRandomSeed();
Random.setSeed(seed);
console.log("Seed:", Random.getSeed());

const vocab = Random.pick(["a sailboat"]);
console.log(vocab);

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
  const NET_LEARNING_RATE = 0.005;
  const NET_SCALE = 30;
  const GAUSSIAN_INPUT_DIMENSION = 2;

  let epochs = 0;

  const outputGamut = Color.sRGBGamut;

  const connection = await createConnection();
  connection.sendPrompt(vocab);
  window.setPrompt = (prompt) => connection.sendPrompt(prompt);

  const background = "#fff";
  const blitter = createBlitter({
    size: TARGET_SIZE,
    background,
    draw: drawFitness,
  });

  const outputCount = 3;
  const inputCount = GAUSSIAN_INPUT_DIMENSION + 2;

  const createNet = () =>
    SirenNetwork({
      inputs: inputCount,
      outputs: outputCount,
      scale: NET_SCALE,
      hiddenLayers: Array(HIDDEN_LAYER_COUNT).fill(HIDDEN_LAYER_SIZE),
    });

  const tmpNet = createNet();
  const netLength = tmpNet.toFlat().length;
  const tempFloat3 = new Float32Array(netLength);

  const colorSigma = 0.1;
  const nodeParams = [
    {
      type: "oklab",
      dimensions: 3,
      sigma: [colorSigma, colorSigma, colorSigma],
      initScale: 0,
    },
    {
      type: "net",
      dimensions: netLength,
      sigma: NET_LEARNING_RATE,
      initData: () => createNet().toFlat(),
    },
  ];

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

  function extractColor(node) {
    const inOKLab = activate(node.oklab);
    const mappedRGB = gamutMapFromLearned(inOKLab);
    return Color.serialize(mappedRGB, Color.sRGB);
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
      solution,
    } = opts;

    const learned = unflatten(solution);
    const { nodes } = learned;

    if (clear) {
      context.clearRect(0, 0, width, height);
      context.fillStyle = background;
      context.fillRect(0, 0, width, height);
    }

    context.lineWidth = 0.01 * width * (fitness ? 2 : 1);
    context.lineJoin = "round";
    context.lineCap = "round";
    context.strokeStyle = "black";

    const dim = Math.min(width, height);
    const padding = 0.1 * dim;

    for (let node of nodes) {
      context.fillStyle = extractColor(node);
      const vertexCount = fitness ? 100 : 1024;

      tempFloat3.set(node.net);
      tmpNet.fromFlat(tempFloat3);

      context.beginPath();
      for (let i = 0; i < vertexCount; i++) {
        const t = i / vertexCount;
        let angle = t * Math.PI * 2;
        if (!fitness && exporting) angle += Random.gaussian(0, 0.02);

        let idx = 0;
        tmpInput[idx++] = Math.cos(angle);
        tmpInput[idx++] = Math.sin(angle);
        tmpInput.set(gaussianInputs, idx);

        tmpNet.forward(tmpInput, tmpOutput);

        const tanhScale = 0.5;
        const scl = 1.2;
        const outU = Math.tanh(tanhScale * tmpOutput[0]) * scl;
        const outV = Math.tanh(tanhScale * tmpOutput[1]) * scl;
        const x = width / 2 + (outU * (width - padding * 2)) / 2;
        const y = height / 2 + (outV * (height - padding * 2)) / 2;
        context.lineTo(x, y);
      }
      context.closePath();
      context.stroke();
    }
  }

  function gamutMapFromLearned(input) {
    return Color.gamutMapOKLCH(
      Color.convert(input, Color.OKLab, Color.OKLCH),
      outputGamut,
      outputGamut.space,
      undefined,
      Color.MapToAdaptiveCuspL,
    );
  }

  function activate(color) {
    const abScale = 0.4;
    return [
      clamp01(sigmoid(color[0])),
      clamp(Math.tanh(color[1]) * abScale, -abScale, abScale),
      clamp(Math.tanh(color[2]) * abScale, -abScale, abScale),
    ];
  }

  function drawFitness(opts) {
    drawSolution({
      ...opts,
      exporting: false,
      background,
      fitness: true,
    });
  }

  function flatten(data) {
    const out = new Float32Array(solutionLength);
    let idx = 0;
    for (let obj of data.nodes) {
      for (let { type, dimensions } of nodeParams) {
        for (let c = 0; c < dimensions; c++) {
          out[idx++] = obj[type][c];
        }
      }
    }
    return out;
  }

  function unflatten(solution, opts = {}) {
    const data = { nodes: [] };
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
    const { bestFitness } = await blitter.tick({ connection, optimizer });
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

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
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
    const v = clamp01(inverseLerp(bounds[0], bounds[1], plot[i]));
    context.lineTo(t * width, (1 - v) * height);
  }
}
