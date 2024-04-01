config {
  module = false
  force = false
}

plugin "google" {
  enabled = true
  version = "0.27.1"
  source  = "github.com/terraform-linters/tflint-ruleset-google"
}

rule "google_container_cluster_invalid_machine_type" {
  enabled = false
}

rule "google_container_node_pool_invalid_machine_type" {
  enabled = false
}

rule "terraform_naming_convention" {
  enabled = true
  format  = "snake_case"
}
