from typing import List

from langchain_core.documents import Document


class Validator:
    def validate(self, ranked_docs: List[Document]):

        top1 = ranked_docs[0][1]

        avg_top3 = sum(
            score
            for _, score in ranked_docs[:3]
        ) / 3

        confidence = (
            0.7 * top1 +
            0.3 * avg_top3
        )

        return confidence > 0.7