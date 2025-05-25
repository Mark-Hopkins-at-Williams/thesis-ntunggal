"""
Takes the original RIME Wubi86 character to keystroke mapping and converts
it into a character to component mapping, writing it to a new yaml file.
"""
key_to_component = {
    'a': '工',
    's': '木',
    'd': '大',
    'f': '土',
    'g': '王',
    'h': '目',
    'j': '日',
    'k': '口',
    'l': '田',
    'm': '山',
    'x': '纟',
    'c': '又',
    'v': '女',
    'b': '子',
    'n': '已',
    'q': '金',
    'w': '人',
    'e': '月',
    'r': '白',
    't': '禾',
    'y': '言',
    'u': '立',
    'i': '水',
    'o': '火',
    'p': '之',
    'z': 'Ⓩ'
}


def convert_wubi_to_component_yaml(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        parsing = False

        for line in fin:
            stripped = line.strip()

            # Preserve header, comments, empty lines
            if not parsing:
                fout.write(line)
                if stripped == "...":
                    parsing = True
                continue
            if not stripped or stripped.startswith('#'):
                fout.write(line)
                continue

            def convert_code(code):
                return ''.join(key_to_component.get(k, f"[{k}]") for k in code)
            
            parts = stripped.split('\t')
            if len(parts) == 4:
                char, simple_code, id, full_code = parts
                new_simple = convert_code(simple_code)
                new_full = convert_code(full_code)
                fout.write(f"{char}\t{new_simple}\t{id}\t{new_full}\n")
            elif len(parts) == 3:
                char, code, id = parts
                new_code = convert_code(code)
                fout.write(f"{char}\t{new_code}\t{id}\n")
            elif len(parts) == 2:
                char, code = parts
                new_code = convert_code(code)
                fout.write(f"{char}\t{new_code}\n")
            else:
                print(f"Malformed line: {line}")
                fout.write(line)  # Leave malformed lines untouched

    print("Done")


if __name__ == "__main__":
    input_path = "/mnt/storage/ntunggal/wubi86.dict.yaml"
    output_path = "wubi86-components.dict.yaml"
    convert_wubi_to_component_yaml(input_path, output_path)