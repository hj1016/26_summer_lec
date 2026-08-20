package com.lecture.rag.lab21rerank;

import com.lecture.rag.lab22.LlmReranker;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Lab 2.1 응용 — PgVectorStore로 넓게 검색한 후보를 LLM으로 재정렬한다.
 * 원본 lab21과 lab22를 수정하지 않고 두 실습의 핵심을 결합한 프로필이다.
 */
@Component
@Profile("lab21-rerank")
public class PgVectorRerankDemo implements CommandLineRunner {

    private static final String DOCUMENT_PATH = "classpath:/docs/manual.pdf";
    private static final String QUERY = "환불은 며칠 안에 가능해?";

    private final VectorStore vectorStore;
    private final ChatModel chatModel;

    public PgVectorRerankDemo(VectorStore vectorStore, ChatModel chatModel) {
        this.vectorStore = vectorStore;
        this.chatModel = chatModel;
    }

    @Override
    public void run(String... args) {
        indexIfEmpty();

        List<Document> plainTop5 = search(5);
        System.out.println("=== 1. PGVector 순수 검색 top-5 ===");
        printPreview(plainTop5);
        System.out.println();

        List<Document> candidates = search(20);
        System.out.printf("=== 2. PGVector 후보 검색: 요청 20개, 실제 %d개 ===%n", candidates.size());

        ChatClient chatClient = ChatClient.builder(chatModel)
                .defaultSystem("항상 숫자만 답하세요.")
                .build();
        LlmReranker reranker = new LlmReranker(chatClient);

        List<LlmReranker.Scored> scored = reranker.scoreAll(QUERY, candidates);
        scored.stream()
                .sorted((a, b) -> Integer.compare(b.score(), a.score()))
                .forEach(result -> System.out.printf(
                        "점수 %2d | %s...%n",
                        result.score(),
                        preview(result.doc(), 80)));
        System.out.println();

        List<Document> rerankedTop5 = scored.stream()
                .sorted((a, b) -> Integer.compare(b.score(), a.score()))
                .limit(5)
                .map(LlmReranker.Scored::doc)
                .toList();

        System.out.println("=== 3. LLM rerank 최종 top-5 ===");
        printPreview(rerankedTop5);
        System.out.println();

        long overlap = plainTop5.stream().filter(rerankedTop5::contains).count();
        System.out.printf("두 top-5에 공통으로 포함된 청크: %d / 5%n", overlap);
    }

    private void indexIfEmpty() {
        if (!search(1).isEmpty()) {
            System.out.println("=== 기존 PGVector 데이터를 사용합니다 ===");
            System.out.println();
            return;
        }

        PagePdfDocumentReader pdfReader = new PagePdfDocumentReader(DOCUMENT_PATH);
        List<Document> documents = pdfReader.get();
        TokenTextSplitter splitter = TokenTextSplitter.builder()
                .withChunkSize(200)
                .build();
        List<Document> chunks = splitter.apply(documents);

        vectorStore.add(chunks);
        System.out.println("=== PGVector가 비어 있어 manual.pdf를 인덱싱했습니다 ===");
        System.out.println("생성된 청크 수: " + chunks.size());
        System.out.println();
    }

    private List<Document> search(int topK) {
        return vectorStore.similaritySearch(
                SearchRequest.builder()
                        .query(QUERY)
                        .topK(topK)
                        .similarityThresholdAll()
                        .build());
    }

    private void printPreview(List<Document> documents) {
        for (Document document : documents) {
            System.out.println("- " + preview(document, 120) + "...");
        }
    }

    private String preview(Document document, int length) {
        String text = document.getText().replaceAll("\\s+", " ");
        return text.substring(0, Math.min(length, text.length()));
    }
}
