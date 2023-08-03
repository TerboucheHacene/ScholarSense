from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from docarray import BaseDoc
from docarray.typing import TorchTensor


@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    authors: List[str]
    categories: List[str]
    primary_category: str
    created: datetime.date
    updated: datetime.date
    obtained: datetime.date
    file_name: str
    doi: Optional[str] = None
    journal_reference: Optional[str] = None
    pdf_url: Optional[str] = None

    def __post_init__(self):
        # http://arxiv.org/abs/2306.12881v1 -> 2306.12881v1
        self.file_name = self.id.rsplit("/", 1)[-1]

    def __repr__(self) -> str:
        return (
            f"Paper(id={self.id}, title={self.title}, abstract={self.abstract[:20]}",
            f"authors={self.authors}, categories={self.categories},",
            f"primary_category={self.primary_category}, doi={self.doi},",
            f"journal_reference={self.journal_reference}, pdf_url={self.pdf_url},",
            f"created={self.created}, updated={self.updated})",
        )

    @classmethod
    def from_json(cls, json_file: str) -> "Paper":
        with open(json_file, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, json_path: str) -> None:
        # convert datetime.datetime to datetime.date
        if isinstance(self.created, datetime.datetime):
            self.created = self.created.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(self.updated, datetime.datetime):
            self.updated = self.updated.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(self.obtained, datetime.date):
            self.obtained = self.obtained.strftime("%Y-%m-%d %H:%M:%S")

        json_file = os.path.join(json_path, self.file_name + ".json")
        with open(json_file, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def to_dict(self) -> Dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, dict: Dict) -> "Paper":
        return cls(**dict)


class DocPaper(BaseDoc):
    id: str
    title: str
    abstract: str
    authors: List[str]
    categories: List[str]
    primary_category: str
    created: str
    updated: str
    obtained: str
    filename: Optional[str]
    pdf_url: Optional[str]
    doi: Optional[str]
    journal_reference: Optional[str]
    embedding: Optional[TorchTensor]

    @classmethod
    def from_json(cls, path: str) -> "DocPaper":
        with open(path, "r") as f:
            data = json.load(f)
        data["id"] = data["id"].rsplit("/", 1)[-1]
        return cls(**data)
