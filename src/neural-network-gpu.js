import lookup from './lookup';
import TrainStream from './train-stream';
import max from './utilities/max';
import mse from './utilities/mse';
import randos from './utilities/randos';
import range from './utilities/range';
import toArray from './utilities/to-array';
import zeros from './utilities/zeros';
import ones from './utilities/ones';
import GPU from 'gpu.js';

/**
 *
 * @param {object} options
 * @constructor
 */
export default class NeuralNetworkGPU {
  constructor(options = {}) {
    Object.assign(this, NeuralNetworkGPU.defaults, options);
    this.hiddenSizes = options.hiddenLayers;
    this.layers = null;
    this.sizes = null;
    this.outputLayer = null;
    this.biases = null; // weights for bias nodes
    this.weights = null;
    this.outputs = null;

    this.forwardPropagate = [];
    this.backwardPropagate = [];
    this.changesPropagate = [];
    this.weightsPropagate = [];
    this.biasesPropagate = [];
    // state for training
    this.deltas = null;
    this.changes = null; // for momentum
    this.errors = null;
    this.gpu = new GPU({mode: 'webgl'});
  }

  /**
   *
   * @param {} sizes
   * @param {Boolean} keepNetworkIntact
   */
  initialize(sizes, keepNetworkIntact) {
    this.sizes = sizes;
    this.outputLayer = this.sizes.length - 1;

    if (!keepNetworkIntact) {
      this.biases = []; // weights for bias nodes
      this.weights = [];
      this.outputs = [];
    }

    // state for training
    this.deltas = [];
    this.changes = []; // for momentum
    this.errors = [];

    for (let layer = 0; layer <= this.outputLayer; layer++) {
      let size = this.sizes[layer];
      this.deltas[layer] = zeros(size);
      this.errors[layer] = zeros(size);
      if (!keepNetworkIntact) {
        this.outputs[layer] = zeros(size);
      }

      if (layer > 0) {
        this.biases[layer] = randos(size);
        
        if (!keepNetworkIntact) {
          this.weights[layer] = new Array(size);
        }
        this.changes[layer] = new Array(size);

        for (let node = 0; node < size; node++) {
          let prevSize = this.sizes[layer - 1];
          if (!keepNetworkIntact) {
            this.weights[layer][node] = randos(prevSize);
          }
          this.changes[layer][node] = zeros(prevSize);
        }
      }
    }
    this.buildRunInput();
    this.buildCalculateDeltas();
    this.buildGetChanges();
    // this.buildChangeWeights();
    this.buildChangeBiases();

  }


  /**
   *
   * @param data
   * @param options
   * @returns {{error: number, iterations: number}}
   */
  train(data, _options = {}) {
    const options = Object.assign({}, NeuralNetworkGPU.trainDefaults, _options);
    data = this.formatData(data);
    let iterations = options.iterations;
    let errorThresh = options.errorThresh;
    let log = options.log === true ? console.log : options.log;
    let logPeriod = options.logPeriod;
    let learningRate = _options.learningRate || this.learningRate || options.learningRate;
    let callback = options.callback;
    let callbackPeriod = options.callbackPeriod;
    let sizes = [];
    let inputSize = data[0].input.length;
    let outputSize = data[0].output.length;
    let hiddenSizes = this.hiddenSizes;
    if (!hiddenSizes) {
      sizes.push(Math.max(3, Math.floor(inputSize / 2)));
    } else {
      hiddenSizes.forEach(size => {
        sizes.push(size);
      });
    }

    sizes.unshift(inputSize);

    sizes.push(outputSize);
    
    this.initialize(sizes, options.keepNetworkIntact);
    
    let error = 1;
    let i;
    for (i = 0; i < 1 && error > errorThresh; i++) {
      let sum = 0;
      for (let j = 0; j < data.length; j++) {
        let err = this.trainPattern(data[j].input, data[j].output, learningRate);
        sum += err;
      }
      error = sum / data.length;

      if (log && (i % 10 == 0)) {
        log('iterations:', i, 'training error:', error);
      }
      if (callback && (i % callbackPeriod == 0)) {
        callback({ error: error, iterations: i });
      }
    }

    return {
      error: error,
      iterations: i
    };
  }

  /**
   *
   * @param input
   * @param target
   * @param learningRate
   */
  trainPattern(input, target, learningRate) {
    learningRate = learningRate || this.learningRate;

    // forward propagate
    this.runInput(input);

    // backward propagate
    this.calculateDeltas(target);

    this.getChanges(learningRate);

    // this.changeBiases(learningRate);

    let error = mse(this.errors[this.outputLayer]);
    return error;
  }
  
  // weights: [Layer1 - undefined, Layer2 - [[0,0], [0,0], [0,0]], Layer3 - [[0,0,0]]];
  // changes: [Layer1 - undefined, Layer2 - [ [0,0], [0,0], [0,0] ], Layer3 - [[0,0,0]]];
  // biases:  [Layer1 - undefined, Layer2 - [0, 0, 0], Layer3 - [0]]
  // outputs: [Layer1 - [0,0], Layer2 - [0, 0, 0], Layer3 - [0]]
  // errors:  [Layer1 - [0,0], Layer2 - [0,0,0], Layer3 - [0] ];
  // deltas:  [Layer1 - [0,0], Layer2 - [0,0,0], Layer3 - [0] ];
  // sizes: [2, 3, 1];

  buildRunInput() {
    function weightedSum(weights, biases, x, inputs) {
        var sum = biases[x];
        for (var k = 0; k < size; k++) {
            sum += weights[x][k] * input[k];
        }
        return 1 / (1 + Math.exp(-sum));
    }
    for(var layer = 1; layer <= this.outputLayer; layer++){
      const kernel = this.gpu.createKernelMap([weightedSum], function(weights, biases, input){
              return weightedSum(weights, biases, gpu_threadX, input);
        }, {
            constants:{
                size: this.sizes[layer - 1]
            }
        }).setDimensions([this.sizes[layer]]);
      this.forwardPropagate[layer] = kernel;
    }
  }

  /**
   *
   * @param input
   * @returns {*}
   */
  runInput(input){
    let output;
    this.outputs[0] = input;
    for (var layer = 1; layer <= this.outputLayer; layer++) {
        this.outputs[layer] = this.forwardPropagate[layer](
        this.weights[layer], this.biases[layer], input
        ).result;
        output = input = this.outputs[layer];
    }
    return output;
  }

  buildCalculateDeltas(target){    

      function calcError(outputs, target) {
          return target[gpu_threadX] - outputs[gpu_threadX];
      }

      function calcDeltas(error, output) {
          return error * output * (1 - output);
      }

      function calcErrorOutput(nextWeights, nextDeltas){
          var error = 0;
          for(var k = 0; k < size; k++){
              error += nextDeltas[k] * nextWeights[k][gpu_threadX];
          }
          return error;
      }

      for(var layer = this.outputLayer; layer >= 0; layer--){
          if(layer == this.outputLayer){
              const kernel = this.gpu.createKernelMap({
                error: calcError, 
                deltas: calcDeltas
              }, function(outputs, target){
                  var output = outputs[gpu_threadX];
                  return calcDeltas(calcError(outputs, target), output);
              }).setDimensions([this.weights[layer].length]);
              this.backwardPropagate[layer] = kernel;
  
          }else{
              const kernel = this.gpu.createKernelMap({
                  error: calcErrorOutput, 
                  deltas: calcDeltas
                }, function(nextWeights, outputs, nextDeltas){
                  var output = outputs[gpu_threadX];
                  return calcDeltas(calcErrorOutput(nextWeights, nextDeltas), output);
              }, {
                  constants: {
                      size: this.deltas[layer + 1].length
                  }
              }).setDimensions([this.sizes[layer]]);
              this.backwardPropagate[layer] = kernel;
          }
      }
  }

  calculateDeltas(target){
      for (var layer = this.outputLayer; layer >= 0; layer--) {
        let output;
        if(layer == this.outputLayer){
            output = this.backwardPropagate[layer](
              this.outputs[layer], 
              target);
        } else {
          output = this.backwardPropagate[layer](
              this.weights[layer + 1],
              this.outputs[layer], 
              this.deltas[layer + 1]);
        }

        this.deltas[layer] = output.result;
        this.errors[layer] = output.error.toArray(this.gpu);
      }
  }

  buildGetChanges(){
    function calcChange(delta, previousOutputs, changes, momentum, learningRate){
        var change = changes[gpu_threadX];
        change = (learningRate * delta * previousOutputs) + (momentum * change);
        return change;
    }

    function addChanges(change, weights){
      var sum = change;
      sum += weights[gpu_threadX]; 
      return sum;
    }

    for (let layer = 1; layer <= this.outputLayer; layer++) {
      const kernel = this.gpu.createKernelMap({
        calcChange: calcChange,
        addChanges: addChanges
      }, function(previousOutputs, deltas, weights, changes, learningRate, momentum){
              var delta = deltas;
              var change = calcChange(delta, previousOutputs, changes, momentum, learningRate);
              addChanges(change, weights);
              return change;
      }, {outputToTexture: false})
      .setDimensions([this.sizes[layer - 1]]);
    
      this.changesPropagate[layer] = kernel;
    }
  }
  
  getChanges(learningRate){
    for (let layer = 1; layer <= this.outputLayer; layer++) {
      let incoming = this.outputs[layer - 1];

      for (let node = 0; node < this.sizes[layer]; node++) {
        let delta = this.deltas[layer][node];

        for (let k = 0; k < incoming.length; k++) {
          let change = this.changes[layer][node][k];

          change = (learningRate * delta * incoming[k])
            + (this.momentum * change);
          
          this.changes[layer][node][k] = change;
          this.weights[layer][node][k] += change;
        }
        this.biases[layer][node] += learningRate * delta;
      }
    }
  }

    // getChanges(){
    // console.log(this.changes[1], 'beforeChanges');
    // for (let layer = 1; layer <= this.outputLayer; layer++) {
      
    //   for(let node = 0; node < this.sizes[layer]; node++){

    //   let output = this.changesPropagate[layer](
    //     this.outputs[layer - 1][node],
    //     this.deltas[layer][node],
    //     this.weights[layer][node],
    //     this.changes[layer][node],
    //     learningRate,
    //     this.momentum
    //   );

    //   this.changes[layer][node] = output.result;
    //   this.weights[layer][node] = output.addChanges.toArray(this.gpu);
    //   }
    
    // }
    //   console.log(this.weights[1][0]);
    //   console.log(this.changes[1], 'changes');
    // }


  buildChangeBiases(){
      for(let layer = 1; layer <= this.outputLayer; layer++){
      const kernel = this.gpu.createKernelMap(function(deltas, learningRate){
          var delta = deltas[gpu_threadX];
          return delta * learningRate;
      }).setDimensions([this.sizes[layer]]);
      
      this.biasesPropagate[layer] = kernel;
    }
  }

  changeBiases(learningRate){
    for(let layer = 1; layer <= this.outputLayer; layer++){
      let output = this.biasesPropagate[layer](
        this.deltas[layer],
        learningRate
      );
      this.biases[layer] = output;
    }
  }

  // buildChangeWeights(){
  //     function add(left, right) {
  //        return left[this.thread.y][this.thread.x] * right[this.thread.y][this.thread.x];
  //     }

  //   for(let layer = 1; layer <= this.outputLayer; layer++){
  //     const kernel = this.gpu.createKernelMap([add], function(changes, weights){
  //       return add(changes, weights);
  //     }, {
  //       constants: {
  //         size: this.sizes[layer - 1]
  //       }
  //     }).setDimensions([this.sizes[layer - 1], this.sizes[layer]]);
      
  //     this.weightsPropagate[layer] = kernel;
  //   }
  // }

  // changeWeights(){
  //   for(let layer = 1; layer <= this.outputLayer; layer++){
  //     let output = this.weightsPropagate[layer](
  //       this.changes[layer],
  //       this.weights[layer]
  //     ).result;
  //     this.weights[layer] = output;
  //     // console.log(output[0], layer);
  //   }    
  // }


  /**
   *
   * @param input
   * @returns {*}
   */
  run(input) {
    if (this.inputLookup) {
      input = lookup.toArray(this.inputLookup, input);
    }
    let output = this.runInput(input);

    if (this.outputLookup) {
      output = lookup.toHash(this.outputLookup, output);
    }
    return output;
  }

  /**
   *
   * @param data
   * @returns {*}
   */
  formatData(data) {
    if (data.constructor !== Array) { // turn stream datum into array
      let tmp = [];
      tmp.push(data);
      data = tmp;
    }
    // turn sparse hash input into arrays with 0s as filler
    let datum = data[0].input;
    if (datum.constructor !== Array && !(datum instanceof Float64Array)) {
      if (!this.inputLookup) {
        this.inputLookup = lookup.buildLookup(data.map(value => value['input']));
      }
      data = data.map(datum => {
        let array = lookup.toArray(this.inputLookup, datum.input);
        return Object.assign({}, datum, { input: array });
      }, this);
    }

    if (data[0].output.constructor !== Array) {
      if (!this.outputLookup) {
        this.outputLookup = lookup.buildLookup(data.map(value => value['output']));
      }
      data = data.map(datum => {
        let array = lookup.toArray(this.outputLookup, datum.output);
        return Object.assign({}, datum, { output: array });
      }, this);
    }
    return data;
  }



}

NeuralNetworkGPU.trainDefaults = {
  iterations: 20000,
  errorThresh: 0.005,
  log: false,
  logPeriod: 10,
  learningRate: 0.3,
  callback: null,
  callbackPeriod: 10,
  keepNetworkIntact: false
};

NeuralNetworkGPU.defaults = {
  learningRate: 0.3,
  momentum: 0.1,
  binaryThresh: 0.5,
  hiddenLayers: null
};