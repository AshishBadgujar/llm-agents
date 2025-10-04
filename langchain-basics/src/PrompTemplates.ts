import { ChatPromptTemplate } from '@langchain/core/prompts'

import { ChatOpenAI } from '@langchain/openai'

const model = new ChatOpenAI({
    apiKey: "fake-key", // Set a dummy key if required by LangChain
    modelName: "llama-3.2-3b", // Specify the model name, if needed
    temperature: 0.8,          // Adjust based on your use case
    maxTokens: 700,
    verbose: false
});

async function fromTemplate() {
    const prompt = ChatPromptTemplate.fromTemplate('Write a short description for the following product: {product_name}')
    // const wholePrompt = await prompt.format({
    //     product_name: 'bicycle'
    // })

    const chain = prompt.pipe(model)
    const response = await chain.invoke({
        product_name: 'bicycle'
    })
    console.log(response)
}
fromTemplate()

async function fromMessage() {
    const prompt = ChatPromptTemplate.fromMessages([
        ['system', 'Write a short description for the product provided by the user'],
        ['human', '{product_name}'],
    ])
    const chain = prompt.pipe(model)
    const response = await chain.invoke({
        product_name: 'bicycle'
    })
    console.log(response)
}
fromMessage()