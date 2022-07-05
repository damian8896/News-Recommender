let mainpage = document.querySelector(".articles-container");

function moveRight(){
    mainpage.style.transform = "translate(-100vw)"
    scrollDistance = event.deltay
}

let scrollDistance = 0;

function replaceVerticalScrollByHorizontal(event) {
    if (event.deltaY != 0) {
        scrollDistance -= event.deltaY
        mainpage.style.transform = "translate()"
    }
    return;
}

window.addEventListener('scroll', function(){
    mainpage.style.transform = "translate("
});