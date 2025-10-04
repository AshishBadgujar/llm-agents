import { Ollama, OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'
// to work with ollama embedings -  ollama pull mxbai-embed-large

const llm = new Ollama({
    model: "llama3.2:3b", // Default value
    temperature: 0.7,
    maxRetries: 2,
});

const question = "what is Overlay Networks?"
async function main() {

    const loader = new CheerioWebBaseLoader('https://srivastavayushmaan1347.medium.com/mastering-docker-swarm-a-deep-dive-into-container-orchestration-2a383e8808ec')
    const docs = await loader.load()

    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 200,
        chunkOverlap: 20
    })

    const splittedDocs = await splitter.splitDocuments(docs)


    const vectorStore = new MemoryVectorStore(new OllamaEmbeddings())
    await vectorStore.addDocuments(splittedDocs)

    const retriver = vectorStore.asRetriever({
        k: 2 // number of results
    })

    const results = await retriver._getRelevantDocuments(question)
    const resultDocs = results.map(result => result.pageContent)

    const template = ChatPromptTemplate.fromMessages([
        ['system', 'Answer the use question based on the following context:{context}'],
        ['user', '{input}']
    ])

    const chain = template.pipe(llm)

    const response = await chain.invoke({
        input: question,
        context: resultDocs
    })
    console.log(response)

}
main()