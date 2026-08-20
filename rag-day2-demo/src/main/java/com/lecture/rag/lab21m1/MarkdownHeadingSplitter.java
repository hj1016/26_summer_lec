package com.lecture.rag.lab21m1;

import org.springframework.ai.document.Document;

import java.util.ArrayList;
import java.util.List;

/**
 * Markdown 제목(#, ##, ### ...)을 1차 청크 경계로 사용한다.
 * 제목 아래 내용이 너무 길면 RecursiveCharacterSplitter로 다시 나눠
 * 문서 구조를 우선 보존하면서 최대 청크 크기도 제한한다.
 */
public class MarkdownHeadingSplitter {

    private final StructureBasedSplitter structureSplitter;
    private final RecursiveCharacterSplitter fallbackSplitter;
    private final int maxChunkChars;

    public MarkdownHeadingSplitter(int maxChunkChars) {
        this.maxChunkChars = maxChunkChars;
        this.structureSplitter = StructureBasedSplitter.forMarkdownHeaders();
        this.fallbackSplitter = new RecursiveCharacterSplitter(maxChunkChars);
    }

    public List<Document> split(Document doc) {
        List<Document> sections = structureSplitter.split(doc);
        List<Document> chunks = new ArrayList<>();

        for (Document section : sections) {
            if (section.getText().length() <= maxChunkChars) {
                chunks.add(section);
            } else {
                chunks.addAll(fallbackSplitter.split(section));
            }
        }
        return chunks;
    }
}
