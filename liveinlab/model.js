const tf = require('@tensorflow/tfjs')
const toxicity = require('@tensorflow-models/toxicity')
console.log(tf.version.tfjs);

const threshold = 0.5

toxicity.load(threshold).then((model) => {
  const sentences = ['You are a poopy head!', 'I like turtles', 'Shut up!']

  model.classify(sentences).then((predictions) => {
    // semi-pretty-print results
    console.log(JSON.stringify(predictions, null, 2))
  })
})


