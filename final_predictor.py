import json
import csv
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn
from bs4 import BeautifulSoup
from datetime import datetime, timedelta


class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.fc = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def get_data(url):
    response = requests.get(url)
    text = response.text
    soup = BeautifulSoup(text, 'html.parser')
    element = soup.find('div', class_="col-md-12")
    script = element.find('script')

    if script:
        script_content = script.text

        date_start_index = script_content.find("[", script_content.find("categories: [") + 1)
        date_end_index = script_content.find("]", date_start_index) + 2

        total_cases_start_idx = script_content.find("[", script_content.find("data: [") + 1)
        total_cases_end_idx = script_content.find("]", total_cases_start_idx) + 2

        second_series_cat = script_content[date_start_index:date_end_index]
        second_series_content = script_content[total_cases_start_idx:total_cases_end_idx]

        try:
            dates = json.loads(second_series_cat)
            total_cases = json.loads(second_series_content)
        except json.JSONDecodeError:
            print("Error decoding JSON data.")
            return None
        return dates, total_cases
    else:
        return None


def write_files(data, last_date):
    start_date = datetime.strptime(last_date, "%b %d, %Y")
    next_dates = [start_date + timedelta(days=i) for i in range(2, 9)]
    rows = []
    new_cases = []
    formatted_dates = []

    for i in range(1, len(data)):
        new_cases.append(int(round(data[i] - data[i - 1], 0)))

    for i in range(len(next_dates)):
        new_case = str(new_cases[i])
        formatted_date = next_dates[i].strftime("%d %b, %Y")
        formatted_dates.append(formatted_date)
        rows.append([formatted_date, new_case])

    fields = ['date', 'new_cases']

    with open('./new_case.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)

    fig = plt.figure(figsize=(10, 5))
    plt.bar(formatted_dates, new_cases, width=0.4)
    plt.xlabel("Date")
    plt.ylabel("No. of Cases")
    plt.title("Daily Cases Over 7-Day Period")
    plt.savefig('./new_case.png')
    plt.show()


if __name__ == "__main__":
    load = False

    link = "https://www.worldometers.info/coronavirus/country/south-africa/#graph-cases-daily"
    covid_dates, covid_total_cases = get_data(link)
    size = len(covid_total_cases)
    epochs = 100000
    lr = 0.0005

    true_y = [[covid_total_cases[i]] for i in range(size)]
    true_y = np.array(true_y)
    mean = np.mean(true_y)
    std_dev = np.std(true_y)
    normalized_y = (true_y - mean) / std_dev
    x = [[i] for i in range(size)]
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(normalized_y, dtype=torch.float32)

    model = RegressionModel()
    if load:
        model.load_state_dict(torch.load("./models/model.pth"))
    else:
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            prediction = model(x)

            loss = loss_fn(prediction, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

        torch.save(model.state_dict(), "./models/model.pth")

    # Testing fit against data
    # test_prediction = model(x).detach().numpy()
    # test_prediction = (test_prediction * std_dev) + mean
    #
    # plt.plot(x, true_y)
    # plt.plot(x, test_prediction)
    # plt.show()

    new_x = torch.tensor([[i] for i in range(size, size + 8)], dtype=torch.float32)
    prediction = model(new_x).detach().numpy()
    prediction = (prediction * std_dev) + mean
    write_files(prediction.flatten(), covid_dates[len(covid_dates) - 1])
