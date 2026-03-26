// Chess Master 9001 — Web UI

let boardWidget = null;
let game = new Chess();
let gameHistory = [];
let valueHistory = [];
let isThinking = false;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    initBoard();
    newGame();
});

function initBoard() {
    const config = {
        draggable: true,
        position: 'start',
        pieceTheme: '/static/img/chesspieces/wikipedia/{piece}.png',
        onDragStart: onDragStart,
        onDrop: onDrop,
        onSnapEnd: onSnapEnd,
    };
    boardWidget = Chessboard('board', config);
}

function onDragStart(source, piece, position, orientation) {
    if (game.game_over()) return false;
    if (isThinking) return false;
    // Only allow white pieces (human plays white)
    if (piece.search(/^b/) !== -1) return false;
    if (game.turn() !== 'w') return false;
}

function onDrop(source, target) {
    // Check if it's a promotion
    const piece = game.get(source);
    let promotion = undefined;
    if (piece && piece.type === 'p' && (target[1] === '8' || target[1] === '1')) {
        promotion = 'q'; // Auto-promote to queen
    }

    const move = game.move({
        from: source,
        to: target,
        promotion: promotion
    });

    if (move === null) return 'snapback';

    // Build UCI string
    let uci = source + target;
    if (promotion) uci += promotion;

    setStatus('<span class="spinner"></span>Bot is thinking...');
    isThinking = true;

    // Send to server
    fetch('/api/move', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({move: uci})
    })
    .then(r => r.json())
    .then(data => {
        isThinking = false;

        if (data.error) {
            game.undo();
            boardWidget.position(game.fen());
            setStatus('Error: ' + data.error);
            return;
        }

        // Apply bot's move
        if (data.bot_move) {
            const from = data.bot_move.substring(0, 2);
            const to = data.bot_move.substring(2, 4);
            const promo = data.bot_move.length > 4 ? data.bot_move[4] : undefined;
            game.move({from: from, to: to, promotion: promo});
            boardWidget.position(game.fen());
        }

        // Update UI
        if (data.game_history) {
            gameHistory = data.game_history;
            updateMoveList();
            updateValueChart();
        }

        if (data.analysis) {
            updateAnalysis(data.analysis);
        }

        if (data.game_over) {
            setStatus('Game over: ' + data.result);
        } else {
            setStatus('Your turn — drag a piece to move');
        }
    })
    .catch(err => {
        isThinking = false;
        setStatus('Error: ' + err.message);
    });
}

function onSnapEnd() {
    boardWidget.position(game.fen());
}

function newGame() {
    fetch('/api/new_game', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({})
    })
    .then(r => r.json())
    .then(data => {
        game = new Chess();
        gameHistory = [];
        valueHistory = [];
        boardWidget.position('start');
        document.getElementById('moveList').innerHTML = '';
        document.getElementById('policyList').innerHTML = '';
        document.getElementById('twoPlyPanel').innerHTML = '<span class="muted">Play a move to see analysis</span>';
        document.getElementById('projectionPanel').innerHTML = '<span class="muted">Play a move to see projections</span>';
        updateValueGauge(0);
        clearValueChart();
        setStatus('Your turn — drag a piece to move');
    });
}

function flipBoard() {
    boardWidget.flip();
}

function setStatus(html) {
    document.getElementById('statusBar').innerHTML = html;
}

// === Analysis Rendering ===

function updateAnalysis(analysis) {
    updateValueGauge(analysis.value);
    updatePolicyList(analysis.policy);

    if (analysis.two_ply) {
        updateTwoPly(analysis.two_ply);
    }
    if (analysis.projection) {
        updateProjection(analysis.projection);
    }
}

function updateValueGauge(value) {
    const gauge = document.getElementById('valueGauge');
    const text = document.getElementById('valueText');

    // value is from bot's perspective (black), so negate for display
    const displayVal = -value;
    const pct = Math.max(-1, Math.min(1, displayVal)) * 50;

    if (pct >= 0) {
        gauge.style.left = '50%';
        gauge.style.width = pct + '%';
        gauge.style.background = '#4ecca3';
    } else {
        gauge.style.left = (50 + pct) + '%';
        gauge.style.width = Math.abs(pct) + '%';
        gauge.style.background = '#e94560';
    }

    text.textContent = (displayVal >= 0 ? '+' : '') + displayVal.toFixed(3);
    text.style.color = displayVal >= 0 ? '#4ecca3' : '#e94560';
}

function updatePolicyList(policy) {
    const container = document.getElementById('policyList');
    if (!policy || policy.length === 0) {
        container.innerHTML = '<span class="muted">No policy data</span>';
        return;
    }

    const maxProb = policy[0].prob;
    let html = '';

    for (let i = 0; i < Math.min(policy.length, 6); i++) {
        const p = policy[i];
        const barWidth = (p.prob / maxProb) * 100;
        html += `
            <div class="policy-row">
                <span class="policy-move">${p.move}</span>
                <div class="policy-bar-bg">
                    <div class="policy-bar" style="width: ${barWidth}%"></div>
                </div>
                <span class="policy-prob">${(p.prob * 100).toFixed(1)}%</span>
            </div>`;
    }

    container.innerHTML = html;
}

function updateTwoPly(data) {
    const container = document.getElementById('twoPlyPanel');
    if (!data || !data.candidates || data.candidates.length === 0) {
        container.innerHTML = '<span class="muted">No two-ply data</span>';
        return;
    }

    let html = '<div style="margin-bottom: 8px; font-size: 0.8rem;">';
    html += `Policy: <strong>${data.one_step_choice}</strong> `;
    html += `Two-ply: <strong>${data.selected}</strong>`;
    html += `<span class="agrees-badge ${data.agrees ? 'agrees-yes' : 'agrees-no'}">`;
    html += data.agrees ? 'agree' : 'disagree';
    html += '</span></div>';

    // Sort by q_value descending
    const sorted = [...data.candidates].sort((a, b) => b.q_value - a.q_value);

    for (const c of sorted.slice(0, 4)) {
        const isBest = c.move === data.selected;
        html += `<div class="two-ply-candidate ${isBest ? 'two-ply-best' : ''}">`;
        html += `<div class="two-ply-header">`;
        html += `<span class="two-ply-move">${c.move}</span>`;
        html += `<span class="two-ply-q">Q=${c.q_value >= 0 ? '+' : ''}${c.q_value.toFixed(4)}</span>`;
        html += `</div>`;

        for (const r of c.replies || []) {
            html += `<div class="two-ply-reply">`;
            html += `  opp <span class="reply-move">${r.move}</span>`;
            html += ` (${(r.prob * 100).toFixed(0)}%)`;
            html += ` V=${r.value >= 0 ? '+' : ''}${r.value.toFixed(3)}`;
            html += `</div>`;
        }

        html += '</div>';
    }

    container.innerHTML = html;
}

function updateProjection(data) {
    const container = document.getElementById('projectionPanel');
    if (!data || !data.lines || data.lines.length === 0) {
        container.innerHTML = '<span class="muted">No projection data</span>';
        return;
    }

    let html = '';

    // Sort by score descending
    const sorted = [...data.lines].sort((a, b) => b.score - a.score);

    for (const line of sorted.slice(0, 4)) {
        const isBest = line.root === data.selected;
        html += `<div class="proj-line ${isBest ? 'proj-line-best' : ''}">`;
        html += `<div class="proj-header">`;
        html += `<span class="proj-root">${line.root}</span>`;
        html += `<span class="proj-score">score=${line.score >= 0 ? '+' : ''}${line.score.toFixed(3)}</span>`;
        html += '</div>';

        // Show move sequence
        const moveStr = line.moves.join(' ');
        html += `<div class="proj-moves">${moveStr}</div>`;

        // Mini value sparkline using SVG
        if (line.values && line.values.length > 1) {
            html += renderSparkline(line.values);
        }

        html += '</div>';
    }

    container.innerHTML = html;
}

function renderSparkline(values) {
    const w = 280, h = 30;
    const padding = 2;
    const minV = Math.min(...values, -0.1);
    const maxV = Math.max(...values, 0.1);
    const range = maxV - minV || 0.1;

    let points = '';
    for (let i = 0; i < values.length; i++) {
        const x = padding + (i / (values.length - 1)) * (w - 2 * padding);
        const y = h - padding - ((values[i] - minV) / range) * (h - 2 * padding);
        points += `${x},${y} `;
    }

    // Zero line
    const zeroY = h - padding - ((0 - minV) / range) * (h - 2 * padding);

    return `<svg class="proj-values" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
        <line x1="${padding}" y1="${zeroY}" x2="${w - padding}" y2="${zeroY}" stroke="#333" stroke-width="0.5" stroke-dasharray="2"/>
        <polyline points="${points}" fill="none" stroke="#4ecca3" stroke-width="1.5"/>
    </svg>`;
}

// === Move List ===

function updateMoveList() {
    const container = document.getElementById('moveList');
    let html = '';

    for (let i = 0; i < gameHistory.length; i += 2) {
        const white = gameHistory[i];
        const black = gameHistory[i + 1];
        const moveNum = Math.floor(i / 2) + 1;

        html += `<span>${moveNum}. ${white.move}`;
        if (black) html += ` ${black.move}`;
        html += '  </span>';
    }

    container.innerHTML = html;
    container.scrollTop = container.scrollHeight;
}

// === Value Chart ===

function updateValueChart() {
    const canvas = document.getElementById('valueChart');
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    if (gameHistory.length < 2) return;

    const values = gameHistory.map(m => m.value);
    const minV = Math.min(...values, -0.3);
    const maxV = Math.max(...values, 0.3);
    const range = maxV - minV || 0.1;

    // Background
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, w, h);

    // Zero line
    const zeroY = h - 5 - ((0 - minV) / range) * (h - 10);
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 0.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, zeroY);
    ctx.lineTo(w, zeroY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Value line
    ctx.strokeStyle = '#4ecca3';
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (let i = 0; i < values.length; i++) {
        const x = 5 + (i / (values.length - 1)) * (w - 10);
        const y = h - 5 - ((values[i] - minV) / range) * (h - 10);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }

    ctx.stroke();

    // Dots
    ctx.fillStyle = '#e94560';
    for (let i = 0; i < values.length; i++) {
        const x = 5 + (i / (values.length - 1)) * (w - 10);
        const y = h - 5 - ((values[i] - minV) / range) * (h - 10);
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
    }
}

function clearValueChart() {
    const canvas = document.getElementById('valueChart');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}
