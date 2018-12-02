import * as tf from '@tensorflow/tfjs';
import "@tensorflow/tfjs-node";
import {generateData} from './data';
import {plotData, plotDataAndPredictions, renderCoefficients} from './ui';

const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));

const numIterations = 75;
const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

function predict(x) {
   return tf.tidy(() => {
      return a.mul(x.pow(tf.scalar(3, 'int32'))).add(b.mul(x.square())).add(c.mul(x)).add(d);
   });
}

function loss(prediction, labels) {
   const error = prediction.sub(labels).square().mean()
   return error;
} 

async function train(xs, ys, numIterations){
   for (var i = 0; i < numIterations; i++) {
      optimizer.minimize(() => {
         const pred = predict(xs);
         return loss(pred, ys);
      });
      
      await tf.nextFrame();
   }
}

async function learnCoefficients() {
   const trueCoefficients = {a: -.8, b: .5, c: .9, d: .4};
   const trainingData = generateData(100, trueCoefficients);
   
   renderCoefficients('#data .coeff', trueCoefficients);
   await plotData('#data .plot', trainingData.xs, trainingData.ys);
   
   renderCoefficients('#random .coeff', {
      a: a.dataSync()[0],
      b: b.dataSync()[0],
      c: c.dataSync()[0],
      d: d.dataSync()[0]
   });
   const initialPreds = predict(trainingData.xs);
   await plotDataAndPredictions('#random .plot', trainingData.xs, trainingData.ys, initialPreds);
   
   await train(trainingData.xs, trainingData.ys, numIterations);
   renderCoefficients('#trained .coeff', {
      a: a.dataSync()[0],
      b: b.dataSync()[0],
      c: c.dataSync()[0],
      d: d.dataSync()[0]
   });
   const finalPreds = predict(trainingData.xs);
   await plotDataAndPredictions('#random .plot', trainingData.xs, trainingData.ys, finalPreds);
   
   initialPreds.dispose(); 
   finalPreds.dispose(); 
}

learnCoefficients();

// const tf = require('@tensorflow/tfjs');
// //import {generateData} from './data';
// //import {plotData, plotDataAndPredictions, renderCoefficients} from './ui';



// const a = tf.variable(tf.scalar(Math.random()));
// const b = tf.variable(tf.scalar(Math.random()));
// const c = tf.variable(tf.scalar(Math.random()));
// const d = tf.variable(tf.scalar(Math.random()));


// const numIterations = 75;
// const learningRate = 0.5;
// const optimizer = tf.train.sgd(learningRate);


// function predict(x) {
//    return tf.tidy(() => {
//       return a.mul(x.pow(tf.scalar(3, 'int32'))).add(b.mul(x.square())).add(c.mul(x)).add(d);
//    });
// }

// function loss(prediction, labels) {
//    const error = prediction.sub(labels).square().mean()
//    return error;
// } 

// async function train(xs, ys, numIterations){
//    for (var i = 0; i < numIterations; i++) {
//       optimizer.minimize(() => {
//          const pred = predict(xs);
//          return loss(pred, ys);
//       });
      
//       await tf.nextFrame();
//    }
// }

// async function learnCoefficients() {
//    const trueCoefficients = {a: -.8, b: .5, c: .9, d: .4};
//    const trainingData = generateData(100, trueCoefficients);
   
//    renderCoefficients('#data .coeff', trueCoefficients);
//    await plotData('#data .plot', trainingData.xs, trainingData.ys);
   
//    renderCoefficients('#random .coeff', {
//       a: a.dataSync()[0],
//       b: b.dataSync()[0],
//       c: c.dataSync()[0],
//       d: d.dataSync()[0]
//    });
//    const initialPreds = predict(trainingData.xs);
//    await plotDataAndPredictions('#random .plot', trainingData.xs, trainingData.ys, initialPreds);
   
//    await train(trainingData.xs, trainingData.ys, numIterations);
//    renderCoefficients('#trained .coeff', {
//       a: a.dataSync()[0],
//       b: b.dataSync()[0],
//       c: c.dataSync()[0],
//       d: d.dataSync()[0]
//    });
//    const finalPreds = predict(trainingData.xs);
//    await plotDataAndPredictions('#random .plot', trainingData.xs, trainingData.ys, finalPreds);
   
//    initialPreds.dispose(); 
//    finalPreds.dispose(); 
// }

// learnCoefficients();

// //export function generateData(numPoints, coeff, sigma = 0.04){
// function generateData(numPoints, coeff, sigma = 0.04){
//    return tf.tidy(() => {
//       const [a, b, c, d] = [
//          tf.scalar(coeff.a), tf.scalar(coeff.b), tf.scalar(coeff.c), tf.scalar(coeff.d)
//          ];
         
//       const xs = tf.randomUniform([numPoints], -1, 1);
      
//       const three = tf.scalar(3, 'int32');
      
//       const ys = a.mul(xs.pow(three)).add(b.mul(xs.square())).add(c.mul(xs)).add(d).add(tf.randomNormal([numPoints], 0, sigma));
      
//       const ymin = ys.min(); 
//       const ymax = ys.max(); 
//       const yrange = ymax.sub(ymin);
//       const ysNormalized = ys.sub(ymin).div(yrange);
      
//       return {
//          xs,
//          ys: ysNormalized
//       };
//    });
// }

// const renderChart = require('vega-embed');
// //import renderChart from 'vega-embed';

// //export async function plotData(container, xs, ys) {
// async function plotData(container, xs, ys) {
//    const xvals = await xs.data(); 
//    const yvals = await ys.data();
   
//    const values = Array.from(yvals).map((y, i) => {
//       return {'x': xvals[i], 'y': yvals[i]};
//    }); 
   
//    const spec = {
//       '$schema': 'https://vega.github.io/schema/vega-lite/v2.json', 
//       'width': 300,
//       'height': 300,
//       'data': {'values': values},
//       'mark': 'point',
//       'encoding': {
//          'x': {'field': 'x', 'type': 'quantitative'},
//          'y': {'field': 'y', 'type': 'quantitative'}
      
//       }
//    };
//    return renderChart(container, spec, {actions: false}); 
// }
// //export async function plotDataAndPredictions(container, xs, ys, preds) {
// async function plotDataAndPredictions(container, xs, ys, preds) {
//    const xvals = await xs.data(); 
//    const yvals = await ys.data();
//    const predVal = await preds.data();
   
//    const values = Array.from(yvals).map((y, i) => {
//       return {'x': xvals[i], 'y': yvals[i], pred: predVal[i]};
//    });
   
//    const spec = {
//       '$schema': 'https://vega.github.io/schema/vega-lite/v2.json', 
//       'width': 300,
//       'height': 300,
//       'data': {'values': values},
//       'layer': [
//          {
//          'mark': 'point',
//          'encoding': {
//          'x': {'field': 'x', 'type': 'quantitative'},
//          'y': {'field': 'y', 'type': 'quantitative'}
      
//          }
//          },
//          {
//             'mark': 'line',
//             'encoding': {
//                'x': {'field': 'x', 'type': 'quantitative'},
//                'y': {'field': 'y', 'type': 'quantitative'},
//                'color': {'value': 'tomato'}
//             },
//          }
//       ]
//    };
//    return renderChart(container, spec, {actions: false});
// }

// //export function renderCoefficients(container, coeff) {
// function renderCoefficients(container, coeff) {
//    document.querySelector(container).innerHTML = 
//    `<span>a=${coeff.a.toFixed(3)}, b=${coeff.b.toFixed(3)},
//    c=${coeff.c.toFixed(3)}, d=${coeff.d.toFixed(3)}</span>`;
// }