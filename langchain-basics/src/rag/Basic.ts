import { Ollama, OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";

// to work with ollama embedings -  ollama pull mxbai-embed-large

const llm = new Ollama({
    model: "llama3.2:3b", // Default value
    temperature: 0.7,
    maxRetries: 2,
});


const myData = [
    "My name is ashish",
    "My name is bob",
    "my favorite food is pizza",
    "my favorite food is pasta"
]

const question = "what are my favorite foods?"
async function main() {
    console.log("creating a vector store...")
    const vectorStore = new MemoryVectorStore(new OllamaEmbeddings())
    console.log("adding data in store as documents...")
    await vectorStore.addDocuments(myData.map(
        content => new Document({ pageContent: content })
    ))

    console.log("creating retriver...")
    const retriver = vectorStore.asRetriever({
        k: 2 // number of results
    })

    console.log("getting relevent result based on questions from vector db...")
    const results = await retriver._getRelevantDocuments(question)
    const resultDocs = results.map(result => result.pageContent)

    console.log("Building chat template...")
    const template = ChatPromptTemplate.fromMessages([
        ['system', 'Answer the use question based on the following context:{context}'],
        ['user', '{input}']
    ])

    const chain = template.pipe(llm)

    console.log("invoking the model...")
    const response = await chain.invoke({
        input: question,
        context: resultDocs
    })
    console.log(response)

}
main()