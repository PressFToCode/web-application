document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    
    form.addEventListener('submit', function(event) {
        const fileInput = document.getElementById('file');
        const file = fileInput.files[0];
        
        if (!file || !file.name.endsWith('.csv')) {
            alert('Пожалуйста, выберите файл в формате .csv');
            event.preventDefault();
        }
        
        const trainStartDate = document.getElementById('train_start_date').value;
        const trainEndDate = document.getElementById('train_end_date').value;
        const forecastStartDate = document.getElementById('forecast_start_date').value;
        const forecastEndDate = document.getElementById('forecast_end_date').value;
        
        if (new Date(trainStartDate) > new Date(trainEndDate)) {
            alert('Дата начала обучения не может быть позже даты окончания обучения');
            event.preventDefault();
        }
        
        if (new Date(forecastStartDate) > new Date(forecastEndDate)) {
            alert('Дата начала прогноза не может быть позже даты окончания прогноза');
            event.preventDefault();
        }
    });
});
