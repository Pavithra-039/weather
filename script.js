function predictTemperature() {
    const year = document.getElementById("year").value;
    const result = document.getElementById("result");

    if (!year) {
        result.innerHTML = "❌ Please enter a valid year!";
        return;
    }

    let predictedTemp = 27 + Math.random() * 5;
    predictedTemp = predictedTemp.toFixed(2);

    result.innerHTML = `🌡️ Predicted Temperature for ${year}: <b>${predictedTemp} °C</b>`;
}
