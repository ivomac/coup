from pettingzoo_coup.env.action import action_space
from pettingzoo_coup.env.state import observation_space
from pettingzoo_coup.env.tabulate import to_markdown


def make_observation_space_table():
    records = observation_space(num_players=6, max_card_count=3)

    new_records = []

    keymap = {
        "idx": "Index Range",
        "size_desc": "Array Length",
        "type": "Array Type",
        "desc": "Description",
        "max": "Value Range",
        "scope": "Scope",
    }

    bit_floor = 0
    for record in records:
        record["max"] = f"0 - {record['max']}"

        bit_ceil = bit_floor + record["size"] - 1
        record["idx"] = f"{bit_floor} - {bit_ceil}"
        bit_floor = bit_ceil + 1

        new_records.append({v: record[k] for k, v in keymap.items()})

    print(to_markdown(new_records))


def make_action_space_table():
    acts = action_space(num_players=6)

    act_info = []

    for i, (act, tgt) in enumerate(acts):
        act_info.append({"Bit": i, "Action": act.name, "Target": tgt})

    print(to_markdown(act_info))


if __name__ == "__main__":
    make_observation_space_table()
    print()
    make_action_space_table()
