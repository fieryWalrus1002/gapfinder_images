import uuid
import csv

def generate_random_uuids(length):
    # Generate a list of random UUIDs
    uuids = [str(uuid.uuid4()) for _ in range(length)]  # Convert UUIDs to strings
    return uuids

if __name__ == "__main__":
    length = 10  # Specify the desired length of the UUID list
    random_uuids = generate_random_uuids(length)

    # Write UUIDs to a CSV file
    with open('random_uuids.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['UUID'])  # Write header row
        for uuid in random_uuids:
            writer.writerow([uuid])
