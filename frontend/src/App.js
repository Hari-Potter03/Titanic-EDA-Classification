import React, { useState } from 'react';

function App() {
  const [formData, setFormData] = useState({
    Pclass: 3,
    Sex: "male",
    Age: 25,
    Fare: 50,
    Embarked: "S",
    Cabin: "U",
    Name: "Braund, Mr. Owen Harris",
    SibSp: 0,
    Parch: 0,
    Ticket: "A/5 21171"
  });

  const [prediction, setPrediction] = useState(null);

  const handleChange = e => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async e => {
    e.preventDefault();
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData)
    });
    const data = await response.json();
    setPrediction(data);
  };

  return (
    <div style={{ padding: "2rem" }}>
      <h2>Titanic Survival Predictor</h2>
      <form onSubmit={handleSubmit}>
        <label>Pclass:
          <select name="Pclass" value={formData.Pclass} onChange={handleChange}>
            <option value="1">1</option><option value="2">2</option><option value="3">3</option>
          </select>
        </label><br/>
        <label>Sex:
          <select name="Sex" value={formData.Sex} onChange={handleChange}>
            <option value="male">male</option><option value="female">female</option>
          </select>
        </label><br/>
        <label>Age: <input name="Age" type="number" value={formData.Age} onChange={handleChange} /></label><br/>
        <label>Fare: <input name="Fare" type="number" value={formData.Fare} onChange={handleChange} /></label><br/>
        <label>Embarked:
          <select name="Embarked" value={formData.Embarked} onChange={handleChange}>
            <option value="C">C</option><option value="Q">Q</option><option value="S">S</option>
          </select>
        </label><br/>
        <label>Cabin: <input name="Cabin" type="text" value={formData.Cabin} onChange={handleChange} /></label><br/>
        <label>Name: <input name="Name" type="text" value={formData.Name} onChange={handleChange} /></label><br/>
        <label>SibSp: <input name="SibSp" type="number" value={formData.SibSp} onChange={handleChange} /></label><br/>
        <label>Parch: <input name="Parch" type="number" value={formData.Parch} onChange={handleChange} /></label><br/>
        <label>Ticket: <input name="Ticket" type="text" value={formData.Ticket} onChange={handleChange} /></label><br/>
        <button type="submit">Predict</button>
      </form>
      {prediction && (
        <div style={{ marginTop: "1rem" }}>
          <h3>Prediction:</h3>
          <p>Survival Probability: {(prediction.survival_probability * 100).toFixed(2)}%</p>
          <p>Predicted Outcome: {prediction.survived === 1 ? "Survived" : "Did not survive"}</p>
        </div>
      )}
    </div>
  );
}

export default App;
