from agents.graph import get_graph

graph = get_graph()

state = {
    "question": "What is the Laplace Transform?"
}

result = graph.invoke(state)

print(result)