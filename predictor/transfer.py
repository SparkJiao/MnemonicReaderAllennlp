import json
from sys import argv


def main(input_file_path, output_file_path):
    with open(input_file_path) as f:
        prediction_file = list()
        for line in f:
            prediction_file.append(json.loads(line))
        outputs = list()
        for output in prediction_file:
            story_id = output["id"]
            answer_text = output["best_span_str"]
            yesno = output["yesno"]
            turn_id = output["turn_id"]
            if yesno == 'y':
                answer_text = 'yes'
            elif yesno == 'n':
                answer_text = 'no'
            pred = {'id': story_id, 'turn_id': turn_id, 'answer': answer_text}
            outputs.append(pred)
        with open(output_file_path, 'w') as file:
            json.dump(outputs, file, indent=4)
        print("transfer complete")


if __name__ == '__main__':
    input_file_path = argv[1]
    output_file_path = argv[2]
    main(input_file_path=input_file_path, output_file_path=output_file_path)
