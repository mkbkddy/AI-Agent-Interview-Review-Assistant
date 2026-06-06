from utils.knowledge_manager import search_knowledge, get_knowledge_list, generate_prompt_from_knowledge, search_knowledge_by_vector

__name__ = "__main__"
results = search_knowledge_by_vector("SpringBoot 是什么？", max_results=5)
print(results)
