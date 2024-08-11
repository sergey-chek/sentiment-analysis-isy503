'use strict'

const API_ENDPOINT = "http://127.0.0.1:8000/analyse-sentiment";

let checkedReviews = [];

function showModal() {
    document.getElementById('modal').classList.add('active');
}

function closeModal() {
    document.getElementById('modal').classList.remove('active');
}

async function analyzeSentiment() {
    const review = document.getElementById('review').value;

    // Check if the textarea is empty
    if (!review.trim()) {
        showModal();
        return;
    }

    const analyzeButton = document.getElementById('analyzeButton');
    const nextReviewButton = document.getElementById('nextReviewButton');
    const positiveBlock = document.getElementById('positiveBlock');
    const negativeBlock = document.getElementById('negativeBlock');
    const neutralBlock = document.getElementById('neutralBlock');
    const reviewTextarea = document.getElementById('review');

    analyzeButton.disabled = true;
    reviewTextarea.disabled = true;
    reviewTextarea.classList.add('disabled-textarea');


    try {
        // add loading spinner
        analyzeButton.innerHTML = `<div class="w-60 spinner"></div>Analyzing...`;

        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ review: review.trim() }),
        });

        const data = await response.json();
        const result = data.sentiment;

        switch (result) {
            case 'positive':
                positiveBlock.classList.add('highlight-positive');
                positiveBlock.classList.remove('hidden');
                break;
            case 'negative':
                negativeBlock.classList.add('highlight-negative');
                negativeBlock.classList.remove('hidden');
                break;
            case 'neutral':
                neutralBlock.classList.add('highlight-neutral');
                neutralBlock.classList.remove('hidden');
                break;
            default:
                alert('An error occurred while analyzing sentiment. Please try again.');
                break;
        }

        analyzeButton.innerHTML = 'Analyze';
        analyzeButton.classList.add('hidden');
        nextReviewButton.classList.remove('hidden');

        const reviewToLog = review.length > 250 ? review.substring(0, 250) + '...' : review;

        const checkedReview = {
            datetime: new Date().toLocaleString(),
            review: reviewToLog,
            sentiment: result,
        };
        checkedReviews.push(checkedReview);

    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while analyzing sentiment. Please try again later.');
        analyzeButton.innerHTML = 'Analyze';
        analyzeButton.disabled = false;
        reviewTextarea.disabled = false;
        reviewTextarea.classList.remove('disabled-textarea');
    }
}

function resetForm() {
    const reviewTextarea = document.getElementById('review');
    reviewTextarea.value = '';
    reviewTextarea.disabled = false;
    reviewTextarea.classList.remove('disabled-textarea');

    document.getElementById('analyzeButton').disabled = false;
    document.getElementById('analyzeButton').classList.remove('hidden');
    document.getElementById('nextReviewButton').classList.add('hidden');

    document.getElementById('positiveBlock').classList.add('hidden');
    document.getElementById('negativeBlock').classList.add('hidden');
    document.getElementById('neutralBlock').classList.add('hidden');

    if (checkedReviews.length > 0) {
        document.getElementById('reviewsTable').innerHTML = '';
        document.getElementById('reviewsBlock').classList.remove('hidden');
        checkedReviews.forEach((checkedReview) => {
            let sentimentColor = '';
            switch (checkedReview.sentiment) {
                case 'positive':
                    sentimentColor = 'green';
                    break;
                case 'negative':
                    sentimentColor = 'red';
                    break;
                case 'neutral':
                    sentimentColor = 'gray';
                    break;
                default:
                    sentimentColor = 'gray';
                    break;
            };
            const template = `
                <div class="grid grid-cols-3 gap-4 mb-4">
                    <div class="col-span-1 text-gray-400 text-sm whitespace-nowrap">${checkedReview.datetime}</div>
                    <div class="col-span-1 text-gray-400 text-sm text-left">${checkedReview.review}</div>
                    <div class="col-span-1"><span class="bg-${sentimentColor}-500 text-white text-sm font-semibold px-2 py-1 rounded-md w-40 inline-block text-center">${checkedReview.sentiment}</span></div>
                </div>`;
            document.getElementById('reviewsTable').insertAdjacentHTML('afterbegin', template);
        });
    }
}

function clearReviews() {
    checkedReviews = [];
    document.getElementById('reviewsTable').innerHTML = '';
    document.getElementById('reviewsBlock').classList.add('hidden');
}