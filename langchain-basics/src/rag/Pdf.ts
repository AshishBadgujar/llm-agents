import { Ollama, OllamaEmbeddings } from "@langchain/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { Chroma } from "@langchain/community/vectorstores/chroma"
// to work with ollama embedings -  ollama pull mxbai-embed-large

const llm = new Ollama({
    model: "llama3.2:3b", // Default value
    temperature: 0.7,
    maxRetries: 2,
});

const question = "summarize the document"
async function main() {

    const loader = new PDFLoader('request-journey.pdf', { splitPages: false })
    const docs = await loader.load()

    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 500, // Add a reasonable chunk size
        chunkOverlap: 50, // Add some overlap between chunks
    })

    const splittedDocs = await splitter.splitDocuments(docs)


    const vectorStore = await Chroma.fromDocuments(splittedDocs, new OllamaEmbeddings(), {
        collectionName: 'rag-collection',
        url: 'http://localhost:8000'
    })
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