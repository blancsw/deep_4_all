from typing import Literal, Optional

import datasets
from torch import Generator
from torch.utils.data import Dataset

from oasst_data import ExportMessageNode, visit_threads_depth_first
from oasst_data import read_dataset_message_trees
from oasst_data.formating import Utterance, DatasetEntrySft, Role


class ListDataset(Dataset):
    def __init__(self, data: list):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_oasst_export(
        hf_dataset_name: Optional[str] = "OpenAssistant/oasst2",
        lang: str = "fr,en,de,es,it",
        top_k: Optional[int] = None,
        manual_seed: int = 287631038922,
        mode: Literal["sft", "rm", "rl"] = "sft",
) -> ListDataset:
    """
    :param hf_dataset_name: Optional[str] = "OpenAssistant/oasst2"
        The name of the Hugging Face dataset to load the oasst export from. Defaults to "OpenAssistant/oasst2".

    :param lang: str = "fr,en,de,es,it"
        Comma-separated list of languages to include in the export. Or "*" for all languages. Defaults to "fr,en,de,es,it".

    :param top_k: Optional[int] = None
        Top k thread to export, 1 mean export top thread quality. Defaults to None, which includes all replies.

    :param manual_seed: int = 287631038922
        The manual seed to use for the random number generator.

    :param mode: Literal["sft", "rm", "rl"] = "sft"
        The mode for processing the export. Must be one of "sft" (supervised fine-tuning), "rm" (Reward Models), or "rl" (Reinforcement Learning).

    :return: ListDataset
        The processed oasst export data as a ListDataset, containing either DatasetEntrySft or tuples of prefix and replies
        based on the specified mode.

    """
    if mode not in ("sft", "rm", "rl"):
        raise ValueError(f"Unknown dataset mode: {mode}")

    all_langs = True if lang == '*' else False
    lang_codes: list[str] = lang.split(",")

    generator = Generator()
    generator.manual_seed(manual_seed)

    tree_iter = read_dataset_message_trees(hf_dataset_name, split="train+validation")
    threads_per_tree = []
    for tree in tree_iter:
        tree_lang = tree.prompt.lang
        if tree.tree_state != "ready_for_export" or not tree.prompt.review_result or (
                not all_langs and tree_lang not in lang_codes):
            continue

        if mode in ("sft", "rm"):
            if tree.tree_state != "ready_for_export":
                continue
        elif mode == "rl":
            if tree.tree_state not in ("ready_for_export", "prompt_lottery_waiting"):
                continue

        # extract all threads up to last assistant reply
        threads: list[list[ExportMessageNode]] = []

        def thread_filter(thread: list[ExportMessageNode]) -> bool:
            if any(m.deleted or m.synthetic for m in thread):
                return False

            if top_k is not None:
                for i, m in enumerate(thread):
                    if m.role == "assistant":
                        if m.rank is None:
                            if i > 0 and len(thread[i - 1].replies) > 1:
                                return False
                        elif m.rank >= top_k:
                            return False
            return True

        def leaf_filter(thread: list[ExportMessageNode]) -> bool:
            if mode == "sft":
                # in SFT mode `not thread[-1].replies` finds nodes without children (leaves).
                # We are interested in those which are role='assistant' but some trees don't end on assistant nodes
                # but have prompter leaves .. we want to use those trees too .. e.g. remove the last prompter message(s)
                # so that they end with assistant. The `thread[-2].replies[0] == thread[-1]` check makes sure that only
                # the FIRST prompter reply is added .. e.g. the parent does not appear multiple times and we can use
                # pop() to remove superfluous prompter leaf node later.
                return (
                        len(thread) > 1
                        and not thread[-1].replies
                        and (thread[-1].role == "assistant" or thread[-2].replies[0] == thread[-1])
                        and thread_filter(thread)
                )
            elif mode == "rm":
                # for reward models we use thread-fragments ending on prompter messages as prefix and
                # their (ranked) replies as possible continuations.
                if thread[-1].replies is None:
                    return False
                return (
                        thread[-1].role == "prompter"
                        and len([r for r in thread[-1].replies if r.rank is not None]) > 1
                        and thread_filter(thread)
                )
            elif mode == "rl":
                # during rl we are interested in all possible prefixes ending in prompter messages
                return thread[-1].role == "prompter" and not any(m.deleted or m.synthetic for m in thread)

            raise RuntimeError()

        visit_threads_depth_first(tree.prompt, visitor=threads.append, predicate=leaf_filter)
        if mode == "sft":
            for t in threads:
                if t[-1].role == "prompter":
                    t.pop()

        threads_per_tree.append(threads)

    def process_thread(thread: list[ExportMessageNode]):
        if mode == "sft":
            # ensure roles are strictly alternating between prompter and assistant
            assert all(m.role == "prompter" for m in thread[0::2]) and all(m.role == "assistant" for m in thread[1::2])
            conversation: list[Utterance] = [
                Utterance(
                    text=m.text,
                    role=Role.prompter if m.role == "prompter" else Role.assistant,
                    lang=m.lang,
                    quality=m.get_label_value("quality"),
                    humor=m.get_label_value("humor"),
                    creativity=m.get_label_value("creativity"),
                )
                for m in thread
            ]
            return DatasetEntrySft(conversation=conversation)
        elif mode == "rm":
            prefix = [m.text for m in thread]
            replies = [r for r in thread[-1].replies if r.role == "assistant" and r.rank is not None]
            replies = sorted(replies, key=lambda r: r.rank)
            replies = [r.text for r in replies]
            return prefix, replies
        elif mode == "rl":
            return ([m.text for m in thread],)

        raise RuntimeError()

    # split on tree basis, messages from same tree must not end up in different splits
    trees = ListDataset(threads_per_tree)
    # Flatten the dataset
    return ListDataset([process_thread(thread) for tree_threads in trees for thread in tree_threads])


if __name__ == "__main__":
    dataset = load_oasst_export(top_k=1)
    formated = []
    langs = []
    for conversation in dataset.data:
        formated.append(conversation.get_formatted())
        langs.append(conversation.conversation[0].lang)

    ds = datasets.Dataset.from_dict({"conversation": formated, "langs": langs})
    ds.push_to_hub("blancsw/oasst2_top1_chat_format", commit_message="Add Lang column")
