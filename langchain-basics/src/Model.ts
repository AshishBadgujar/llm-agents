import { Ollama } from "@langchain/ollama";

const llm = new Ollama({
    model: "llama3.2:3b", // Default value
    temperature: 0,
    maxRetries: 2,
});


async function main() {
    const response = await llm.invoke(
        'Hello'
    )
    console.log(response)

    const response2 = await llm.batch([
        "Hello",
        "I am Ashish"
    ])
    console.log(response2)

    const response3 = await llm.stream("Hello")
    for await (const chunk of response3) { console.log(chunk) }
}
main()