let mainpage = document.querySelector(".articles-container");
let articles = document.getElementsByClassName("articles-outer");
let thumbs = document.getElementsByClassName("thumbs");

let works = true;

for(let i = 0; i < thumbs.length; i++){
    thumbs[i].addEventListener("mousedown", event => {
        works = false;
        thumbs[i].style.transform = "translateY(5px)";
        $(thumbs[i]).toggleClass('selected');
    });
    thumbs[i].addEventListener("mouseup", event => {
        thumbs[i].style.transform = "translateY(0px)";
        sleep(100).then(() => {
            works = true;
        });
    });
}

for(let i = 0; i < articles.length; i++){
    articles[i].addEventListener("click", event => {
        if(works){
            window.open("https://www.nbcnews.com/news/us-news/highland-park-shooting-suspects-littered-red-flags-rcna36766", '_blank').focus();
        }
    });
}

let scroll_enabled = true

function moveRight() {
    console.log("works");
    mainpage.style.transform = "translate(-100vw)";
    mainpage.style.transition = "transform 1s ease-in";
    scrollDistance = -window.innerWidth;
    scroll_enabled = false;
    setTimeout(function () {
        scroll_enabled = true;
    }, 1000);
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

let scrollDistance = 0;

window.addEventListener("wheel", event => {
    if (scroll_enabled) {
        if (scrollDistance <= 0 && scrollDistance >= -mainpage.clientWidth) {
            mainpage.style.transition = "transform 0s";
            scrollDistance -= event.deltaY * 0.5;
            mainpage.style.transform = "translate(" + scrollDistance + "px)";
        } else if (scrollDistance > 0) {
            scroll_enabled = false;
            scrollDistance = 0;
            mainpage.style.transition = "transform 0.5s";
            mainpage.style.transform = "translate(0px)";
            sleep(1000).then(() => {
                scroll_enabled = true;
                scrollDistance = 0;
            });
        } else if (scrollDistance < -mainpage.clientWidth) {
            scroll_enabled = false;
            scrollDistance = -mainpage.clientWidth + 1;
            mainpage.style.transition = "transform 0.5s";
            mainpage.style.transform = "translate(" + scrollDistance + "px)";
            sleep(1000).then(() => {
                scroll_enabled = true;
                scrollDistance = -mainpage.clientWidth + 1;
            });
        }
    }
});