import tf from '@tensorflow/tfjs-node';


async function trainModel(inputxs, outputys) {
    const model = tf.sequential()

    // primeira camada da rede:
    // entrada de 7 posicoes (idade normalizada, 3 cores one-hot, 3 localizacoes one-hot)
    // 80 neurônios 
    // função de ativação ReLU age como um filtro esencial para a rede aprender padrões complexos, introduzindo não-linearidade.
    // se a informação que chegou nesse neuronio é positiva, ele passa adiante, se for negativa, ele bloqueia a informação, ou seja, não passa nada adiante.

    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }))

    // saída da rede: 3 neurônios (premium, medium, basic) e função de ativação softmax que transforma as saídas em probabilidades, indicando a probabilidade de cada categoria ser a correta para a entrada dada.
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))
    
    // compilamos o modelo usando o otimizador Adam, que é eficiente para muitos tipos de problemas de aprendizado de máquina, e a função de perda categoricalCrossentropy, que é adequada para problemas de classificação multiclasse.
    // adam é um treinador que ajusta os pesos da rede para minimizar a perda, ou seja, melhorar a precisão das previsões.

    // loss: categoricalCrossentropy é uma função de perda usada para problemas de classificação multiclasse, onde as classes são mutuamente exclusivas. Ela mede a diferença entre as distribuições de probabilidade previstas pelo modelo e as distribuições reais (labels). O objetivo do treinamento é minimizar essa perda, o que significa que o modelo está aprendendo a fazer previsões mais precisas.
    // compara o que o modelo acha que é a resposta correta (as probabilidades previstas) com o que realmente é a resposta correta (as labels one-hot encoded). Quanto mais próximas as previsões estiverem das labels, menor será a perda, indicando que o modelo está aprendendo a classificar corretamente os dados de entrada.
    // exemplo classico: classificaçao de imagens, recomendações, categorizações de usuários, etc.

    // metrics: ['accuracy'] é uma métrica usada para avaliar o desempenho do modelo durante o treinamento e a avaliação. A acurácia mede a proporção de previsões corretas em relação ao total de previsões feitas. Durante o treinamento, a acurácia é calculada para cada época (passo de treinamento) e pode ser usada para monitorar o progresso do modelo. Uma acurácia mais alta indica que o modelo está fazendo previsões mais precisas.
    
 
 
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    })

    // Treinamos o modelo usando os dados de entrada (inputxs) e as labels (outputys) por um número definido de épocas (epochs). Durante o treinamento, o modelo ajusta seus pesos para minimizar a perda e melhorar a acurácia. O processo
    await model.fit(
            inputxs, 
            outputys, 
            {   verbose: 0, // para não mostrar o progresso do treinamento no console
                epochs: 100, // número de vezes que o modelo verá todo o conjunto de dados durante o treinamento
                shuffle: true, // embaralha os dados a cada época para melhorar o aprendizado e evitar overfitting
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`);
                    }
                }
            }
        )
        return model
}

async function predict(model, pessoas, nome) {
    // transformar o array js para o formato de tensor que o modelo espera
    const tfinput = tf.tensor2d([pessoas]) // transformam  os o array de características em um tensor 2D, onde cada linha representa uma pessoa e cada coluna representa uma característica (idade normalizada, cor one-hot, localização one-hot).

    //faz a predição usando o modelo treinado. O método predict retorna um tensor contendo as probabilidades previstas para cada categoria (premium, medium, basic) com base nas características de entrada fornecidas.
    const prediction = model.predict(tfinput)
    const predArray = await prediction.array() // converte o tensor de previsão para um array JavaScript usando o método array(), que retorna uma promessa que resolve para um array contendo as probabilidades previstas para cada categoria.
    console.log(predArray) // exibe o array de probabilidades previstas no console para verificar os resultados da predição.

    // extrai os dados do tensor de previsão para um array JavaScript usando o método dataSync(), que retorna um array contendo as probabilidades previstas para cada categoria.
    const predictedProbabilities = prediction.dataSync()

    // encontra o índice da categoria com a maior probabilidade prevista usando o método indexOf() em conjunto com Math.max(), que retorna o índice do valor máximo no array de probabilidades previstas.
    const predictedIndex = predictedProbabilities.indexOf(Math.max(...predictedProbabilities))

    // usa o índice encontrado para acessar a categoria correspondente no array labelsNomes, que contém os nomes das categorias (premium, medium, basic). O resultado é a categoria prevista para a pessoa com base nas características fornecidas.
    const predictedCategory = labelsNomes[predictedIndex]

    console.log(`A pessoa ${nome} foi classificada como: ${predictedCategory}`)
}

    // Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

async function main() {
    const model = await trainModel(inputXs, outputYs)
    console.log('Modelo treinado com sucesso!')

    await predict(model, pessoaTensorNormalizado, pessoas.nome)

}

const pessoas = {nome: "Leno", idade: 28, cor: "verde", localizacao: "Curitiba"}
//const pessoas = {nome: "Fernanda", idade: 41, cor: "azul", localizacao: "Acre"}
// Para prever a categoria de uma nova pessoa, precisamos transformar suas características em um vetor de entrada compatível com o modelo. Isso envolve normalizar a idade e aplicar one-hot encoding para as características categóricas (cor e localização)
// normalizando a idade de uma nova pessoa (por exemplo, Leno) para o intervalo [0, 1] 
// exemplo: idade_min=25, idade max=40, idade_normalizada = (idade - idade_min) / (idade_max - idade_min)
const idadeMin = 25
const idadeMax = 40
const idadeNormalizada = (pessoas.idade - idadeMin) / (idadeMax - idadeMin)

const pessoaTensorNormalizado = [   idadeNormalizada,
                                    pessoas.cor === "azul" ? 1 : 0, 
                                    pessoas.cor === "vermelho" ? 1 : 0, 
                                    pessoas.cor === "verde" ? 1 : 0, 
                                    pessoas.localizacao === "São Paulo" ? 1 : 0, 
                                    pessoas.localizacao === "Rio" ? 1 : 0, 
                                    pessoas.localizacao === "Curitiba" ? 1 : 0
                                ]

main()