document.addEventListener("DOMContentLoaded", () => {

    const input = document.getElementById("imageInput");
    const preview = document.getElementById("preview");

    if (input) {

        input.addEventListener("change", function () {

            const file = this.files[0];

            if (file) {

                preview.src = URL.createObjectURL(file);

                preview.style.display = "block";

            }

        });

    }

});