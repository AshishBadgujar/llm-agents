import { StringOutputParser, CommaSeparatedListOutputParser, StructuredOutputParser } from '@langchain/core/output_parsers';
import { ChatPromptTemplate } from '@langchain/core/prompts'

import { ChatOpenAI } from '@langchain/openai'

const model = new ChatOpenAI({
    apiKey: "fake-key", // Set a dummy key if required by LangChain
    modelName: "llama-3.2-3b", // Specify the model name, if needed
    temperature: 0.8,          // Adjust based on your use case
    maxTokens: 700,
    verbose: false
});

async function stringParser() {
    const prompt = ChatPromptTemplate.fromTemplate('Write a short description for the following product: {product_name}')
    // const wholePrompt = await prompt.format({
    //     product_name: 'bicycle'
    // })
    const parser = new StringOutputParser()
    const chain = prompt.pipe(model).pipe(parser)
    const response = await chain.invoke({
        product_name: 'bicycle'
    })
    console.log(response)
}
async function commaSaperatedParser() {
    const prompt = ChatPromptTemplate.fromTemplate('Write a short description for the following product: {product_name}')
    // const wholePrompt = await prompt.format({
    //     product_name: 'bicycle'
    // })
    const parser = new CommaSeparatedListOutputParser()
    const chain = prompt.pipe(model).pipe(parser)
    const response = await chain.invoke({
        product_name: 'bicycle'
    })
    console.log(response)
}
async function structuredParser() {
    const templatePrompt = ChatPromptTemplate.fromTemplate(`
        
        Extracting information from the following phrase.
        formatting instructions: {format_instructions}
        Phrase:{phrase}`)

    const outputParser = StructuredOutputParser.fromNamesAndDescriptions({
        name: 'the name of the person',
        likes: 'what the person likes'
    })
    const chain = templatePrompt.pipe(model).pipe(outputParser)
    const result = await chain.invoke({
        phrase: 'john is likes pineapple pizza',
        format_instructions: outputParser.getFormatInstructions()
    })
    console.log(result)

}

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