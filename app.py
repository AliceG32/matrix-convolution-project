from flask import Flask, request, render_template_string
import json

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Свёртка матриц</title>
    <style>
        body { font-family: Arial; margin: 40px; background: #f0f0f0; }
        .container { max-width: 800px; margin: auto; background: white; padding: 30px; border-radius: 10px; }
        textarea { width: 100%; font-family: monospace; padding: 10px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        .result { margin-top: 20px; padding: 15px; background: #e8f4e8; border-radius: 5px; }
        .error { background: #f8e8e8; color: red; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔢 Вычисление свёртки матриц</h1>
        <form method="post">
            <b>Матрица A (исходная):</b><br>
            <textarea name="matrix_a" rows="4">{{ matrix_a }}</textarea><br><br>
            <b>Ядро свёртки B:</b><br>
            <textarea name="matrix_b" rows="3">{{ matrix_b }}</textarea><br><br>
            <button type="submit"> Вычислить свёртку</button>
        </form>
        {% if result %}
        <div class="result">
            <b>Результат свёртки:</b><br>
            <pre>{{ result }}</pre>
        </div>
        {% endif %}
        {% if error %}
        <div class="result error">
            <b>Ошибка:</b> {{ error }}
        </div>
        {% endif %}
    </div>
</body>
</html>
'''


def convolution(matrix, kernel):
    rows, cols = len(matrix), len(matrix[0])
    krows, kcols = len(kernel), len(kernel[0])
    res_rows, res_cols = rows - krows + 1, cols - kcols + 1
    res = [[0] * res_cols for _ in range(res_rows)]
    for i in range(res_rows):
        for j in range(res_cols):
            total = 0
            for ki in range(krows):
                for kj in range(kcols):
                    total += matrix[i + ki][j + kj] * kernel[ki][kj]
            res[i][j] = total
    return res


@app.route('/', methods=['GET', 'POST'])
def index():
    default_a = "[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]"
    default_b = "[[1,0],[-1,1]]"

    if request.method == 'POST':
        matrix_a_str = request.form.get('matrix_a', default_a)
        matrix_b_str = request.form.get('matrix_b', default_b)

        try:
            a = json.loads(matrix_a_str)
            b = json.loads(matrix_b_str)
            res = convolution(a, b)
            return render_template_string(HTML_TEMPLATE,
                                          matrix_a=matrix_a_str,
                                          matrix_b=matrix_b_str,
                                          result=json.dumps(res, indent=2))
        except Exception as e:
            return render_template_string(HTML_TEMPLATE,
                                          matrix_a=matrix_a_str,
                                          matrix_b=matrix_b_str,
                                          error=str(e))

    return render_template_string(HTML_TEMPLATE,
                                  matrix_a=default_a,
                                  matrix_b=default_b)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)