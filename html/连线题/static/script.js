document.addEventListener('DOMContentLoaded', async () => {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    let currentElement = null; // 用于存储当前选中的元素（无论是 question 还是 option）
    let lines = [];
    let correctRelationships = {};
    let isSubmitted = false;
    let connectedStarts = new Set(); // 新增：用于存储已经连线的开始节点
    let connectedEnds = new Set(); // 新增：用于存储已经连线结束节点

    // 初始化下拉列表
    const unitSelect = document.getElementById('unitSelect');

    // 加载下拉列表选项
    const unitsResponse = await fetch('/api/get_units');
    const units = await unitsResponse.json();
    units.forEach(unit => {
        const option = document.createElement('option');
        option.value = unit;
        option.textContent = unit;
        unitSelect.appendChild(option);
    });

    // 加载题目数据
    async function loadQuestions(unit) {
        try {
            const response = await fetch(`/api/questions?unit=${unit}`);
            if (!response.ok) {
                throw new Error('Failed to load questions');
            }
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error loading questions:', error);
            return null;
        }
    }

    // 初始化界面
    async function initializeInterface(unit) {
        const questionsDiv = document.getElementById('questions');
        const englishOptionsDiv = document.getElementById('englishOptions');
        const optionsDiv = document.getElementById('options');

        // 清空现有内容
        questionsDiv.innerHTML = '';
        englishOptionsDiv.innerHTML = '';
        optionsDiv.innerHTML = '';

        // 加载题目数据
        const data = await loadQuestions(unit);
        if (!data) {
            return;
        }

        // 创建按钮
        data.questions.forEach(q => createButton(q, questionsDiv, 'question', unit));
        data.english_options.forEach(opt => createButton(opt, englishOptionsDiv, 'englishOption', unit));
        data.options.forEach(opt => createButton(opt, optionsDiv, 'option', unit));
    }

    // 监听下拉列表变化
    unitSelect.addEventListener('change', async () => {
        const selectedUnit = unitSelect.value;
        if (selectedUnit) {
            await initializeInterface(selectedUnit);
        }
    });

    // 初始化画布
    function resizeCanvas() {
        canvas.width = document.documentElement.clientWidth;
        canvas.height = document.documentElement.clientHeight;
    }
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // 创建按钮
    function createButton(text, parent, type, unit) {
        const button = document.createElement('div');
        button.className = 'button';
        // button.textContent = text;
        button.dataset.type = type; // 添加类型标识

        if (type === 'question'){
            // 添加音频元素
            const audio = document.createElement('audio');
            audio.src = `/static/output_splits/${unit}/${text}`; // 假设音频文件放在static/audio目录中
            audio.controls = true; // 显示音频控件
            button.appendChild(audio);
        }

        // 添加文本内容
        const textNode = document.createTextNode(text);
        button.appendChild(textNode);

        parent.appendChild(button);

        // 避免重复添加事件监听器
        if (!button.hasEventListener) {
            button.addEventListener('click', () => handleClick(button));
            button.addEventListener('contextmenu', (e) => handleRightClick(e, button)); // 新增右键点击事件
            button.hasEventListener = true;
        }
        return button;
    }

    // 处理点击事件
    function handleClick(button) {
        if (isSubmitted) return;
        if (currentElement) {
            const startType = currentElement.dataset.type;
            const endType = button.dataset.type;

            // 验证连接规则
            const validConnections = {
                'question': ['englishOption'],
                'englishOption': ['question', 'option'],
                'option': ['englishOption']
            };

            if (!validConnections[startType]?.includes(endType)) {
                currentElement.classList.remove('selected');
                currentElement = null;
                return;
            }

            if ((startType==='englishOption' && endType === 'question')
                || (startType==='option' && endType === 'englishOption')){
                [currentElement, button] = [button, currentElement]
            }

            // 检查目标元素是否已经被连线过
            if (connectedStarts.has(currentElement)) {
                currentElement.classList.remove('selected');
                currentElement = null;
                return;
            }

            if (connectedEnds.has(button)) {
                currentElement.classList.remove('selected');
                currentElement = null;
                return;
            }

            // 确定连线方向（questions右连english左，english右连options左）
            const startBtn = currentElement;
            const endBtn = button;
            createLine(startBtn, endBtn);

            currentElement.classList.remove('selected');
            currentElement = null;
        } else {
            if (['question', 'englishOption', 'option'].includes(button.dataset.type)) {
                currentElement = button;
                currentElement.classList.add('selected');
            }
        }
    }

    // 处理右键点击事件（删除连线）
    function handleRightClick(e, button) {
        e.preventDefault(); // 阻止默认的右键菜单
        if (isSubmitted) return;

        // 查找与该按钮相关的连线并删除
        lines = lines.filter(line => {
            if (line.startBtn === button || line.endBtn === button) {
                // 从 connectedStarts 和 connectedEnds 中移除相关按钮
                connectedStarts.delete(line.startBtn);
                connectedEnds.delete(line.endBtn);
                return false;
            }
            return true;
        });
        ctx.reset();
        drawLines(); // 重新绘制画布
    }

    // 创建连线
    function createLine(startBtn, endBtn) {
        const startRect = startBtn.getBoundingClientRect();
        const endRect = endBtn.getBoundingClientRect();

        // 根据元素类型确定连接点
        let startX, startY, endX, endY;
        if (startBtn.dataset.type === 'question') {
            startX = startRect.right;
            startY = startRect.top + startRect.height / 2;
            endX = endRect.left;
            endY = endRect.top + endRect.height / 2;
        } else if (startBtn.dataset.type === 'englishOption') {
            if (endBtn.dataset.type === 'question') {
                startX = startRect.right;
                startY = startRect.top + startRect.height / 2;
                endX = endRect.left;
                endY = endRect.top + endRect.height / 2;
            } else if (endBtn.dataset.type === 'option') {
                startX = startRect.right;
                startY = startRect.top + startRect.height / 2;
                endX = endRect.left;
                endY = endRect.top + endRect.height / 2;
            }
        } else if (startBtn.dataset.type === 'option') {
            startX = startRect.right;
            startY = startRect.top + startRect.height / 2;
            endX = endRect.left;
            endY = endRect.top + endRect.height / 2;
        }

        // 如果是englishOption连options，使用右边连接
        if (startBtn.dataset.type === 'englishOption') {
            startX = startRect.right;
        }

        lines.push({
            startX, startY,
            endX, endY,
            startBtn, endBtn
        });

        // 将已连线的元素添加到集合中
        connectedStarts.add(startBtn);
        connectedEnds.add(endBtn);

        if (!isSubmitted) {
            drawLines();
        }
    }

    // 绘制所有连线
    function drawLines() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.lineWidth = 2;
        ctx.lineCap = 'round';
        lines.forEach(line => {
            if (isSubmitted) {
                ctx.beginPath(); // 开始一个新的路径
                // 检查连线是否正确
                const correctAnswer = correctRelationships[line.startBtn.textContent];
                const isCorrect = correctAnswer === line.endBtn.textContent;
                ctx.strokeStyle = isCorrect ? 'green' : 'red';
            } else {
                ctx.strokeStyle = 'green';
            }
            ctx.moveTo(line.startX, line.startY);
            ctx.lineTo(line.endX, line.endY);
            ctx.stroke();
        });
    }

    // 重置功能
    document.getElementById('resetBtn').addEventListener('click', () => {
        lines = [];
        correctRelationships = {};
        isSubmitted = false;
        ctx.reset();
        connectedStarts = new Set();
        connectedEnds = new Set();
        drawLines(); // 调用 drawLines() 函数来重新绘制画布
        if (currentElement) {
            currentElement.classList.remove('selected');
            currentElement = null;
        }
        document.getElementById('score').textContent = '';
        console.log('Reset: lines cleared, canvas cleared'); // 添加调试信息
    });

    // 提交功能
    document.getElementById('submitBtn').addEventListener('click', async () => {
        isSubmitted = true;
        const userRelationships = lines.map(line => {
            const startText = line.startBtn.textContent;
            const endText = line.endBtn.textContent;
            if (line.startBtn.dataset.type === 'question') {
                return [startText, endText];
            }
            else if (line.startBtn.dataset.type === 'englishOption' && line.endBtn.dataset.type === 'option') {
                return [startText, endText];
            }
            else {
                return [endText, startText];
            }
        });

        const selectedUnit = unitSelect.value; // 获取当前选中的单元

        // 发送所有连线关系
        const response = await fetch('/api/submit', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ relationships: userRelationships, unit: selectedUnit })
        });

        // 处理响应并重新绘制
        const result = await response.json();
        correctRelationships = result.correct_relationships;
        lines = [];

        // 重新创建所有连线（包含english到options的连线）
        userRelationships.forEach(([startText, endText]) => {
            const startBtn = [...document.querySelectorAll('.button')].find(btn => btn.textContent === startText);
            const endBtn = [...document.querySelectorAll('.button')].find(btn => btn.textContent === endText);
            if (startBtn && endBtn) createLine(startBtn, endBtn);
        });

        // 统一绘制所有线条
        drawLines();

        const score = document.getElementById('score');
        score.textContent = `得分: ${result.score}`;
        score.style.fontSize = '1.5em';
        score.style.color = '#333';
    });
});