const TWO_PI = Math.PI * 2;

export function getDefaultPopulationCount(solutionLength) {
  return 4 + Math.floor(3 * Math.log(solutionLength));
}

export default function sNES(opts = {}) {
  const {
    solutionLength,
    alpha = 0.05,
    etaCenter = 1,
    sigmaScaleFactor = () => 1,
    // Population size (number of solutions)
    populationCount = getDefaultPopulationCount(solutionLength),
    // alternative
    // https://people.idsia.ch/~juergen/xNES2010gecco.pdf
    // etaSigma = ((3 / 5) * (3 + Math.log(solutionLength))) /
    //   (solutionLength * Math.sqrt(solutionLength)),
    // https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=b49838b5393b9da9be8789115a850df5a2a64867
    etaSigma = (3 + Math.log(solutionLength)) / (5 * Math.sqrt(solutionLength)),
    random = Math.random,
    mirrored = false,
    momentum = false,
  } = opts;

  const utilities = getWeightVector(populationCount);

  // Initialize center (mu)
  const center = new Float32Array(opts.center ?? solutionLength);

  const indexArray = new Uint16Array(populationCount);
  for (let i = 0; i < populationCount; i++) indexArray[i] = i;
  const baseIndexArray = indexArray.slice();

  const solutions = new Float32Array(populationCount * solutionLength);

  // Initialize std deviation (sigma)
  const sigma = new Float32Array(opts.sigma ?? solutionLength);
  if (opts.sigma == null) sigma.fill(1);
  // apply alpha to sigma
  for (let i = 0; i < sigma.length; i++) sigma[i] = sigma[i] * alpha;
  const gausses = new Float32Array(populationCount * solutionLength);

  const centerStep = new Float32Array(solutionLength);

  return {
    get populationCount() {
      return populationCount;
    },
    get solutionLength() {
      return solutionLength;
    },
    get sigma() {
      return sigma;
    },
    get gaussian() {
      return gausses;
    },
    get center() {
      return center;
    },
    ask,
    getSolutionAt(solutions, index) {
      const off = index * solutionLength;
      return solutions.subarray(off, off + solutionLength);
    },
    tell,
  };

  function ask() {
    const stride = solutionLength;

    if (mirrored) {
      const half = Math.floor(populationCount / 2);

      // Generate mirrored pairs
      for (let i = 0; i < half; i++) {
        const idx1 = i;
        const idx2 = i + half;

        const base1 = idx1 * stride;
        const base2 = idx2 * stride;

        for (let j = 0; j < stride; j++) {
          const g = nextGaussianBoxMuller();

          const mu = center[j];
          const sig = sigma[j] * sigmaScaleFactor(j);

          const k1 = base1 + j;
          const k2 = base2 + j;

          gausses[k1] = g;
          gausses[k2] = -g;

          solutions[k1] = mu + sig * g;
          solutions[k2] = mu - sig * g;
        }
      }

      // If odd, generate one extra (unpaired) sample at the end
      if (populationCount % 2 !== 0) {
        const idx = populationCount - 1;
        const base = idx * stride;

        for (let j = 0; j < stride; j++) {
          const g = nextGaussianBoxMuller();
          const mu = center[j];
          const sig = sigma[j] * sigmaScaleFactor(j);

          const k = base + j;

          gausses[k] = g;
          solutions[k] = mu + sig * g;
        }
      }
    } else {
      // Original non-mirrored code
      for (let i = 0; i < populationCount; i++) {
        const base = i * stride;
        for (let j = 0; j < stride; j++) {
          const k = base + j;
          const g = nextGaussianBoxMuller();
          gausses[k] = g;
          solutions[k] = center[j] + sigma[j] * g * sigmaScaleFactor(j);
        }
      }
    }

    return solutions;
  }

  function nextGaussianBoxMuller(mean = 0, standardDerivation = 1) {
    return (
      mean +
      standardDerivation *
        (Math.sqrt(-2.0 * Math.log(random())) * Math.cos(TWO_PI * random()))
    );
  }

  function insertionSortIndices(indexArray, fitnesses) {
    // Insertion sort, sorting indices in descending order (higher fitness first)
    for (let i = 1; i < indexArray.length; i++) {
      let key = indexArray[i];
      let keyFitness = fitnesses[key];
      let j = i - 1;
      // Shift elements with lower fitness to the right
      while (j >= 0 && fitnesses[indexArray[j]] < keyFitness) {
        indexArray[j + 1] = indexArray[j];
        j--;
      }
      indexArray[j + 1] = key;
    }
  }

  function tell(fitnesses) {
    if (fitnesses.length !== populationCount) {
      throw new Error("Mismatch between population size and fitness values.");
    }

    // Reset the index array, so it goes from 0 ... N - 1
    indexArray.set(baseIndexArray);

    // Sort indices based on fitness
    insertionSortIndices(indexArray, fitnesses);

    // or with builtin sort
    // indexArray.sort((a, b) => fitnesses[b] - fitnesses[a]);

    // Update each parameter dimension
    for (let j = 0; j < solutionLength; j++) {
      let deltaMu = 0;
      let deltaSigma = 0;
      // Sum the utility-weighted noise for this parameter dimension
      for (let i = 0; i < populationCount; i++) {
        const idx = indexArray[i];
        const gaussIndex = idx * solutionLength + j;
        const noise = gausses[gaussIndex];
        deltaMu += utilities[i] * noise;
        deltaSigma += utilities[i] * (noise * noise - 1);
      }

      const step = etaCenter * (sigma[j] * sigmaScaleFactor(j)) * deltaMu;
      // Update center (mu)
      if (!momentum) {
        center[j] += step;
      } else {
        // Apply simple momentum to smooth out random walk behavior
        const newStep = momentum * centerStep[j] + (1 - momentum) * step;
        centerStep[j] = newStep;

        center[j] += newStep;
      }

      // Update sigma: multiplicative update via exponential
      sigma[j] *= Math.exp(0.5 * etaSigma * deltaSigma);
    }

    return indexArray;
  }
}

function getWeightVector(n) {
  // Pre-calculate utilities vector 'us'
  // For i in 0 ... n-1: us[i] = max(0, log(n/2 + 1) - log(1 + i))
  // Then normalize: divide by sum(us) and subtract 1/n
  const us = new Float32Array(n);
  let sumUs = 0;
  for (let i = 0; i < n; i++) {
    let u = Math.max(0, Math.log(n / 2 + 1) - Math.log(1 + i));
    us[i] = u;
    sumUs += u;
  }
  for (let i = 0; i < n; i++) {
    us[i] = us[i] / sumUs - 1 / n;
  }
  return us;
}
