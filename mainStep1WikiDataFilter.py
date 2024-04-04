import bz2
import json
import os
import pydash
import time

def get_records_per_language_using_wiki(wiki_original_file_path, output_directory_path):
    """
    This function searches the records in each language from a Wikimedia dump file.
    It writes the resultant set of IDs to an output file, prompting the user once at the beginning if they wish to overwrite
    the existing files.

    :param languages: The list of languages to filter records.
    :param wiki_original_file_path: The file path of the Wikimedia dump file.
    :param output_directory_path: The directory path to write the resultant sets of IDs.
    :return: None
    """

    # If English file exists, ask user if they want to overwrite since going over .bz2 file will take hours of processing and should be done only when new dataset is introduced
    lang = 'en'
    file_path = f"{output_directory_path}{lang}_wiki_ids.txt"
    if os.path.exists(file_path):
        user_decision = input(
            f"The following file already exist: {file_path}. Do you want to overwrite? (yes/no): ")
        if user_decision.lower() != 'yes':
            print("Aborting the process.")
            return

    print("Calculating file size...")
    # File size
    file_size = os.path.getsize(wiki_original_file_path)

    # Initialize a dictionary to store sets of IDs per language
    id_sets_per_language = {}

    # Start the timer for notifications
    start_time = time.time()

    # Initialize a counter for processed bytes
    bytes_processed = 0

    print(f"print starting to process file {wiki_original_file_path}...")
    # Open the Wikimedia dump file
    with bz2.open(wiki_original_file_path, mode='rt') as f_read:
        # Skip first two bytes: "[\n"
        f_read.read(2)

        # Iterate over each line (record) in the Wikimedia dump file
        for line in f_read:
            try:
                # Parse the record (line) as JSON
                record = json.loads(line.rstrip(',\n'))

                # Iterate over sitelinks in the record
                for sitelink in record.get('sitelinks', {}):
                    # Get the language from the sitelink
                    lang = sitelink.split('wik', 1)[0]

                    # Initialize the language set if it doesn't exist
                    if lang not in id_sets_per_language:
                        id_sets_per_language[lang] = set()

                    # Add the record's ID to the set of the current language
                    id_sets_per_language[lang].add(record.get('id'))

                # add to the count of bytes processed
                bytes_processed += len(line.encode('utf-8'))  # Amount of bytes the line takes

                # Print a notification every minute
                if time.time() - start_time > 60:
                    print(f"Still processing... Processed {bytes_processed} bytes.")
                    start_time = time.time()
            except json.decoder.JSONDecodeError:
                # Skip the current record and continue with the next one if there is a parsing error
                continue

    # Save the sets of IDs to output files per language
    for lang, id_set in id_sets_per_language.items():
        with open(f"{output_directory_path}{lang}_wiki_ids.txt", 'w') as f_write:
            for id_ in id_set:
                f_write.write(f"{id_}\n")

    print("Processing completed.")

def get_common_ids_between_lang_files(lang1_file_path, lang2_file_path):
    """
    This function reads two language files, finds and returns the common IDs.

    :param lang1_file_path: The first language file path.
    :param lang2_file_path: The second language file path.
    :return: The list of common IDs.
    """

    # Initialize two sets to store IDs from the language files
    lang1_ids = set()
    lang2_ids = set()

    # Read the IDs from the first language file
    with open(lang1_file_path, 'r') as f1:
        for id_ in f1:
            lang1_ids.add(id_.strip())

    # Read the IDs from the second language file
    with open(lang2_file_path, 'r') as f2:
        for id_ in f2:
            lang2_ids.add(id_.strip())

    # Find and return the common IDs between the two language files
    return lang1_ids & lang2_ids  # The '&' operator performs set intersection


def filter_wiki_to_manageable_size(set_of_ids, input_filepath, output_filepath):
    """
    This function filters a Wikimedia dump file according to a set of record IDs and saves the filtered records to an
    output file.

    :param set_of_ids: The set of record IDs to filter.
    :param input_filepath: The file path of the Wikimedia dump file.
    :param output_filepath: The output file path to save the filtered records.
    :return: None
    """

    if not os.path.exists(output_filepath):
        # Total number of records to be written
        total_count = len(set_of_ids)

        # Time when the last update was printed
        last_update_time = time.time()

        # Open the input and output files
        with bz2.open(input_filepath, mode='rt') as f_read, bz2.open(output_filepath, mode='wt') as f_write:
            # Skip the first two bytes, that's the opening square bracket
            f_read.read(2)

            # Write the opening square bracket for valid JSON
            f_write.write('[')

            # Counter to keep track of the number of written records
            count = 0

            # Define a flag to control the comma insertion in JSON
            first_record = True

            # Loop over each line (record) of the input file
            for line in f_read:
                line = line.rstrip(',\n')

                # Break the loop if the line is empty or we have reached the total count
                if not line or count >= total_count:
                    break

                try:
                    # Parse the line as a JSON object
                    record = json.loads(line)

                    # Check if the ID of the current record is in the set of ids
                    if record.get('id') in set_of_ids:
                        # If the current record is not the first one, prepend a comma before it for valid JSON syntax
                        if not first_record:
                            f_write.write(',')
                        else:
                            first_record = False

                        # Dump the record as a string and write it to the output file
                        f_write.write(json.dumps(record))

                        count += 1

                        # If more than 60 seconds have passed since the last update,
                        # print an update and record the time of this update.
                        if time.time() - last_update_time > 60:
                            print(f"Written {count} / {total_count} records")
                            last_update_time = time.time()

                except json.JSONDecodeError:
                    print("Error processing line")
                    continue

            # Write the closing square bracket for valid JSON
            f_write.write(']')

        print("Filtering complete!")

if __name__ == '__main__':
    '''
    As of September 2021, the top 20 languages on Wikipedia in terms of the number of articles are as follows:

    1. English ('en') - over 6.3 million articles.
    2. Cebuano ('ceb') - over 5.5 million articles.
    3. Swedish ('sv') - over 3.8 million articles.
    4. German ('de') - over 2.6 million articles.
    5. French ('fr') - over 2.3 million articles.
    6. Dutch ('nl') - over 2 million articles.
    7. Russian ('ru') - over 1.7 million articles.
    8. Italian ('it') - over 1.7 million articles.
    9. Spanish ('es') - over 1.7 million articles.
    10. Polish ('pl') - over 1.4 million articles.
    11. Waray ('war') - over 1.3 million articles.
    12. Vietnamese ('vi') - over 1.2 million articles.
    13. Japanese ('ja') - over 1.2 million articles.
    14. Egyptian Arabic ('arz') - over 1.1 million articles.
    15. Chinese ('zh') - over 1.1 million articles.
    16. Ukrainian ('uk') - over 1.0 million articles.
    17. Catalan ('ca') - over 0.6 million articles.
    18. Persian ('fa') - over 0.7 million articles.
    19. Serbian ('sr') - over 0.6 million articles.
    20. Norwegian (Bokmål) ('no') - over 0.5 million articles.
    '''

    from PathVarAndOther import mainPathWikiWikipediaData
    from PathVarAndOther import step1FolderOut
    output_directory_path = mainPathWikiWikipediaData+step1FolderOut
    if not os.path.exists(output_directory_path):
        os.makedirs(output_directory_path)

    wiki_original_file_path = mainPathWikiWikipediaData+'latest-all.json.bz2'
    get_records_per_language_using_wiki(wiki_original_file_path, output_directory_path)

    lang1 = 'ru'
    lang1_file_path = f"{output_directory_path}{lang1}_wiki_ids.txt"
    lang2 = 'en'
    lang2_file_path = f"{output_directory_path}{lang2}_wiki_ids.txt"
    set_of_ids = get_common_ids_between_lang_files(lang1_file_path, lang2_file_path)

    wiki_output_filepath = f'{mainPathWikiWikipediaData}{lang1}_{lang2}.json'
    filter_wiki_to_manageable_size(set_of_ids, wiki_original_file_path, wiki_output_filepath)
